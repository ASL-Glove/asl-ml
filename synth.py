"""Generate FAKE glove sessions in the exact format collect.py records.

Purpose: exercise the whole pipeline (preprocess -> train -> export -> firmware)
before the glove exists. Accuracy on synthetic data means nothing about real
accuracy; it only proves the plumbing works.

    python synth.py                      # 3 wearers x 2 sessions into data/synthetic/
    python synth.py --wearers 2 --sessions 1 --reps 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import ROOT, load_config


def class_prototypes(cfg: dict, rng: np.random.Generator) -> dict:
    """One fixed 'handshape' per class: flex levels, contacts, gravity direction."""
    n_a, n_d = len(cfg["channels"]["analog"]), len(cfg["channels"]["digital"])
    protos = {}
    for c in cfg["classes"]:
        g = rng.normal(size=3)
        protos[c] = {
            "flex": rng.uniform(0.05, 0.95, n_a),
            "contacts": rng.random(n_d) < 0.3,
            "gravity": g / np.linalg.norm(g),
        }
    # Pairs that share a handshape and differ mainly by orientation (real ASL confusions).
    for a, b in [("K", "P"), ("G", "Q"), ("H", "U"), ("I", "J"), ("D", "Z")]:
        if a in protos and b in protos:
            protos[b]["flex"] = protos[a]["flex"].copy()
            protos[b]["contacts"] = protos[a]["contacts"].copy()
    return protos


def segment_frames(label, proto, n, cfg, sess, rng):
    """Return a DataFrame of n raw frames for one held sign (or 'none')."""
    ch = cfg["channels"]
    n_a = len(ch["analog"])
    t = np.arange(n) / cfg["sample_rate_hz"]
    if label == "none":
        # Hand relaxing / moving between signs: slow random drift.
        flex = np.clip(0.3 + np.cumsum(rng.normal(0, 0.02, (n, n_a)), axis=0), 0, 1)
        contacts = np.zeros((n, len(ch["digital"])), bool)
        grav = np.tile(rng.normal(size=3), (n, 1))
        grav /= np.linalg.norm(grav, axis=1, keepdims=True)
        gyro = rng.normal(0, 40, (n, 3))
    else:
        flex = proto["flex"] + sess["flex_offset"] + rng.normal(0, 0.03, (n, n_a))
        contacts = np.tile(proto["contacts"], (n, 1))
        grav = proto["gravity"] + sess["grav_jitter"] + rng.normal(0, 0.03, (n, 3))
        gyro = rng.normal(0, 5, (n, 3))
        if label in cfg["motion_classes"]:
            # J = hooked twist, Z = zig-zag: distinct rotation waveforms.
            f = 1.2 if label == "J" else 2.0
            gyro[:, 0] += 180 * np.sin(2 * np.pi * f * t)
            gyro[:, 2] += 120 * np.sign(np.sin(2 * np.pi * f * t)) if label == "Z" else 0
        grav /= np.linalg.norm(grav, axis=1, keepdims=True)
    flex = np.clip(flex, 0, 1)
    lo, hi = sess["true_min"], sess["true_max"]
    raw_a = np.clip(lo + flex * (hi - lo) + rng.normal(0, 8, flex.shape), 0, cfg["adc_max"]).round().astype(int)
    df = pd.DataFrame(raw_a, columns=ch["analog"])
    for i, c in enumerate(ch["digital"]):
        df[c] = contacts[:, i].astype(int)
    acc = grav + rng.normal(0, 0.02, grav.shape)
    for i, c in enumerate(ch["accel"]):
        df[c] = acc[:, i].round(3)
    for i, c in enumerate(ch["gyro"]):
        df[c] = gyro[:, i].round(2)
    return df


def make_session(cfg, protos, reps, rng):
    n_a = len(cfg["channels"]["analog"])
    true_min = rng.uniform(500, 1200, n_a)
    sess = {
        "true_min": true_min,
        "true_max": true_min + rng.uniform(1200, 2400, n_a),
        "flex_offset": rng.normal(0, 0.04, n_a),   # glove sits slightly differently each time
        "grav_jitter": rng.normal(0, 0.08, 3),
    }
    rate = cfg["sample_rate_hz"]
    parts, seg = [], 0
    letters = [c for c in cfg["classes"] if c != "none"]
    for _ in range(reps):
        for c in rng.permutation(letters):
            hold = 2.5 if c in cfg["motion_classes"] else 2.0
            for label, secs in [(c, hold), ("none", 1.5)]:
                df = segment_frames(label, protos.get(label), int(secs * rate), cfg, sess, rng)
                df.insert(0, "segment", seg)
                df.insert(0, "label", label)
                parts.append(df)
                seg += 1
    rec = pd.concat(parts, ignore_index=True)
    rec.insert(0, "t_ms", (np.arange(len(rec)) * 1000 // rate).astype(int))
    # Calibration as collect.py would measure it: close to, but not exactly, the true range.
    calib = {"min": (true_min + rng.normal(0, 20, n_a)).round(1).tolist(),
             "max": (sess["true_max"] + rng.normal(0, 20, n_a)).round(1).tolist(),
             "source": "synthetic"}
    return rec, calib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--wearers", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--reps", type=int, default=4, help="repetitions of each letter per session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    protos = class_prototypes(cfg, rng)
    for w in range(args.wearers):
        for s in range(args.sessions):
            rec, calib = make_session(cfg, protos, args.reps, rng)
            d = Path(args.out) / f"synth_wearer{w}" / f"session{s}"
            d.mkdir(parents=True, exist_ok=True)
            rec.to_csv(d / "recording.csv", index=False)
            (d / "calib.json").write_text(json.dumps(calib, indent=2))
            print(f"wrote {d}  ({len(rec)} frames)")


if __name__ == "__main__":
    main()
