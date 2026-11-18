"""Preprocessing shared by training (Python) and the glove firmware (C).

Every formula here has a twin in the firmware's inference.cpp. If you change
one, change the other and regenerate preprocess_config.h with export.py.

Per-frame feature vector (NUM_FEATURES = 16 with the default config):
    analog  : clip((raw - calib_min) / (calib_max - calib_min), 0, 1)
    digital : 0.0 or 1.0
    accel   : clip(g / accel_range_g, -1, 1)
    gyro    : clip(dps / gyro_range_dps, -1, 1)

Session layout on disk (written by collect.py or synth.py):
    data/raw/<wearer>/<session>/recording.csv   columns: t_ms,label,segment,<channels...>
    data/raw/<wearer>/<session>/calib.json      {"min": [...], "max": [...]} per analog channel
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: Path | str = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ch = cfg["channels"]
    cfg["csv_columns"] = ch["analog"] + ch["digital"] + ch["accel"] + ch["gyro"]
    cfg["num_features"] = len(cfg["csv_columns"])
    cfg["adc_max"] = (1 << cfg["adc_bits"]) - 1
    return cfg


def default_calib(cfg: dict) -> dict:
    n = len(cfg["channels"]["analog"])
    return {"min": [0.0] * n, "max": [float(cfg["adc_max"])] * n}


def calib_from_frames(raw_analog: np.ndarray, cfg: dict) -> dict:
    """Min/max per analog channel from open-hand + fist frames (shape [N, n_analog]).

    Uses the 2nd/98th percentiles so a single noisy sample doesn't stretch the range.
    """
    lo = np.percentile(raw_analog, 2, axis=0)
    hi = np.percentile(raw_analog, 98, axis=0)
    # Guard against a dead/unwired channel (zero span would divide by zero).
    span = np.maximum(hi - lo, 1.0)
    return {"min": lo.round(1).tolist(), "max": (lo + span).round(1).tolist()}


def frames_to_features(df: pd.DataFrame, calib: dict, cfg: dict) -> np.ndarray:
    ch = cfg["channels"]
    a = df[ch["analog"]].to_numpy(np.float32)
    lo = np.asarray(calib["min"], np.float32)
    hi = np.asarray(calib["max"], np.float32)
    a = np.clip((a - lo) / np.maximum(hi - lo, 1.0), 0.0, 1.0)
    d = df[ch["digital"]].to_numpy(np.float32)
    acc = np.clip(df[ch["accel"]].to_numpy(np.float32) / cfg["accel_range_g"], -1.0, 1.0)
    gyr = np.clip(df[ch["gyro"]].to_numpy(np.float32) / cfg["gyro_range_dps"], -1.0, 1.0)
    return np.concatenate([a, d, acc, gyr], axis=1)


def make_windows(features: np.ndarray, labels: np.ndarray, segments: np.ndarray, cfg: dict):
    """Slide a window over each labeled segment. Windows never cross segment boundaries,
    so every window has exactly one label.

    Letter segments are trimmed first (config trim_start_s / trim_end_s): right after
    "GO!" the hand is still moving INTO the sign, and at the end it may start relaxing
    early. Those frames are labeled as the letter but don't look like it, so we drop
    them. 'none' segments are not trimmed: transitions are exactly what 'none' should
    contain. Trimming happens here (not at recording time), so it can be changed later
    without re-recording.
    """
    W, S = cfg["window_frames"], cfg["stride_frames"]
    rate = cfg["sample_rate_hz"]
    trim_start = int(round(cfg.get("trim_start_s", 0.0) * rate))
    trim_end = int(round(cfg.get("trim_end_s", 0.0) * rate))
    class_idx = {c: i for i, c in enumerate(cfg["classes"])}
    X, y = [], []
    for seg in np.unique(segments):
        idx = np.flatnonzero(segments == seg)
        label = labels[idx[0]]
        if label not in class_idx:
            continue  # letter dropped from config.classes
        if label != "none":
            idx = idx[trim_start:len(idx) - trim_end]
        for start in range(0, len(idx) - W + 1, S):
            X.append(features[idx[start:start + W]])
            y.append(class_idx[label])
    if not X:
        return np.zeros((0, W, cfg["num_features"]), np.float32), np.zeros((0,), np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64)


@dataclass
class Session:
    wearer: str
    name: str
    path: Path
    df: pd.DataFrame
    calib: dict


def load_sessions(data_dir: Path | str, cfg: dict) -> list[Session]:
    sessions = []
    for rec in sorted(Path(data_dir).glob("*/*/recording.csv")):
        calib_path = rec.parent / "calib.json"
        calib = json.loads(calib_path.read_text()) if calib_path.exists() else default_calib(cfg)
        df = pd.read_csv(rec)
        missing = [c for c in cfg["csv_columns"] + ["label", "segment"] if c not in df.columns]
        if missing:
            raise ValueError(f"{rec}: missing columns {missing}; was it recorded with a different config?")
        sessions.append(Session(rec.parent.parent.name, rec.parent.name, rec.parent, df, calib))
    return sessions


def session_windows(s: Session, cfg: dict):
    feats = frames_to_features(s.df, s.calib, cfg)
    return make_windows(feats, s.df["label"].astype(str).to_numpy(), s.df["segment"].to_numpy(), cfg)


def _c_float_array(vals) -> str:
    return "{" + ", ".join(f"{float(v):.4f}f" for v in vals) + "}"


def emit_header(cfg: dict, path: Path | str, calib: dict | None = None,
                input_scale: float = 1.0, input_zero_point: int = 0) -> None:
    """Write preprocess_config.h for the firmware. Generated; do not hand-edit."""
    calib = calib or default_calib(cfg)
    ch = cfg["channels"]
    labels = "".join("-" if c == "none" else c for c in cfg["classes"])
    text = f"""// AUTO-GENERATED by asl-ml/export.py from config.yaml. Do not edit by hand.
// Regenerate: python export.py   (then copy into the firmware folder)
#pragma once

#define SAMPLE_RATE_HZ        {cfg['sample_rate_hz']}
#define WINDOW_FRAMES         {cfg['window_frames']}
#define NUM_FEATURES          {cfg['num_features']}
#define NUM_ANALOG            {len(ch['analog'])}
#define NUM_DIGITAL           {len(ch['digital'])}
#define ADC_BITS              {cfg['adc_bits']}
#define ACCEL_RANGE_G         {cfg['accel_range_g']:.1f}f
#define GYRO_RANGE_DPS        {cfg['gyro_range_dps']:.1f}f
#define NUM_CLASSES           {len(cfg['classes'])}
#define CONFIDENCE_THRESHOLD  {cfg['confidence_threshold']:.2f}f
#define AGREEMENT_WINDOWS     {cfg['agreement_windows']}
#define INFERENCE_EVERY       {cfg['inference_every_frames']}

// Model input quantization (int8). 1.0 / 0 means "no model exported yet".
#define INPUT_SCALE           {input_scale:.8f}f
#define INPUT_ZERO_POINT      {int(input_zero_point)}

// Class index -> character. '-' = none (no sign / transition).
static const char CLASS_LABELS[NUM_CLASSES + 1] = "{labels}";

// Default per-channel calibration (raw ADC counts). Replace per wearer.
static const float CALIB_MIN[NUM_ANALOG] = {_c_float_array(calib['min'])};
static const float CALIB_MAX[NUM_ANALOG] = {_c_float_array(calib['max'])};
"""
    Path(path).write_text(text)
