"""Record labeled training data from the glove over USB serial.

The firmware must be built with DATA_COLLECTION_MODE set to 1.

    python collect.py --list-ports
    python collect.py --port /dev/cu.usbmodem14101 --wearer cayden --session s1
    python collect.py --port /dev/cu.usbmodem14101 --wearer cayden --session s1 --self-paced
    python collect.py --port /dev/cu.usbmodem14101 --wearer cayden --session s1 --letters A B C --reps 3
    python collect.py --port fake --wearer test --session s0      # no glove: synthetic stream

Each session:
  1. Calibration: hold hand OPEN (3 s), then make a FIST (3 s) -> calib.json.
     You see the result and can redo it before continuing.
  2. For each letter x reps (shuffled): countdown, hold the sign, then relax.
     The relax period is recorded as the 'none' class.

Keys while recording (no Enter needed):
    u      undo: throw away the most recent sign and redo it next.
           Press again to go further back. Works mid-sign too (discards that attempt).
    space  pause / resume (a sign in progress is discarded and redone after resuming)
    q      quit and save everything kept so far (Ctrl+C also works)
    Enter  (only with --self-paced) start the next sign when you're ready

Output: data/raw/<wearer>/<session>/recording.csv + calib.json + meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np
import pandas as pd

from preprocess import ROOT, calib_from_frames, load_config

# Below this many ADC counts between open hand and fist, a flex channel is probably
# unplugged or badly placed (12-bit ADC: 0..4095).
MIN_CALIB_SPAN = 100


# ----------------------------------------------------------------------------
# Reading frames from the glove
# ----------------------------------------------------------------------------
class Recorder:
    """Reads frames continuously on a background thread (so the serial buffer never
    overflows while the main thread is waiting on the user), and tags every frame with
    the segment currently being recorded. A segment = one attempt at one sign, or one
    relax period. Keeping segments separate is what makes undo exact: we delete whole
    segments, never a slice of time that could cut a sign in half."""

    def __init__(self, source, cfg):
        self.source = source
        self.n = cfg["num_features"]
        self.rows = []                 # [t_ms, label, segment, ...channels]
        self.bad_lines = 0
        self.good_lines = 0
        self._label, self._seg, self._active = None, None, False
        self._next_seg = 0
        self._lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            line = self.source.readline()
            if not line or line.startswith("#"):
                continue  # firmware comment/header lines start with '#'
            parts = line.strip().split(",")
            if len(parts) != self.n + 1:  # t_ms + channels
                self.bad_lines += 1
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                self.bad_lines += 1
                continue
            self.good_lines += 1
            with self._lock:
                if self._active:
                    self.rows.append([int(vals[0]), self._label, self._seg] + vals[1:])

    def start(self, label) -> int:
        """Begin tagging incoming frames with `label`; returns the new segment id."""
        with self._lock:
            self._label, self._seg, self._active = label, self._next_seg, True
            self._next_seg += 1
            return self._seg

    def stop(self) -> int:
        """Stop recording; returns how many frames the segment got."""
        with self._lock:
            self._active = False
            return sum(1 for r in self.rows if r[2] == self._seg)

    def discard(self, segments):
        segs = set(segments)
        with self._lock:
            self.rows = [r for r in self.rows if r[2] not in segs]

    def frames(self, segment):
        with self._lock:
            return [r for r in self.rows if r[2] == segment]


class SerialSource:
    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self.ser.reset_input_buffer()

    def readline(self):
        return self.ser.readline().decode("ascii", errors="replace")


class FakeSource:
    """Synthetic 50 Hz stream so the collection flow can be tried without the glove."""

    def __init__(self, cfg):
        self.cfg, self.t0 = cfg, time.time()
        self.rng = np.random.default_rng()

    def readline(self):
        time.sleep(1 / self.cfg["sample_rate_hz"])
        ch = self.cfg["channels"]
        a = self.rng.integers(800, 3000, len(ch["analog"]))
        d = self.rng.integers(0, 2, len(ch["digital"]))
        rest = self.rng.normal(0, 0.5, 6)
        t = int((time.time() - self.t0) * 1000)
        return ",".join(map(str, [t, *a, *d, *rest.round(3)])) + "\n"


# ----------------------------------------------------------------------------
# Single-key input (no Enter) on macOS/Linux and Windows
# ----------------------------------------------------------------------------
class Keys:
    """Puts the terminal in 'cbreak' mode so single key presses arrive immediately
    and aren't echoed. Always restores the terminal on exit, even after a crash."""

    def __enter__(self):
        self._old = None
        self._win = os.name == "nt"
        if not self._win and sys.stdin.isatty():
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def _get(self, timeout):
        """One key, or None if nothing was pressed within `timeout` seconds."""
        if self._win:
            import msvcrt
            end = time.time() + timeout
            while time.time() < end:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
                time.sleep(0.02)
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], max(timeout, 0))
        if not r:
            return None
        ch = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if ch == "":                    # stdin closed (e.g. piped input ran out)
            time.sleep(timeout)
            return None
        return ch

    def wait(self, seconds, accept):
        """Wait up to `seconds`; return the first key in `accept`, or None on timeout."""
        end = time.time() + seconds
        while (left := end - time.time()) > 0:
            ch = self._get(left)
            if ch is None:
                continue
            ch = "\n" if ch == "\r" else ch.lower()
            if ch in accept:
                return ch
        return None

    def wait_forever(self, accept):
        while True:
            ch = self.wait(3600, accept)
            if ch:
                return ch


# ----------------------------------------------------------------------------
# Session steps
# ----------------------------------------------------------------------------
def calibrate(rec, cfg):
    """Open hand + fist -> per-channel min/max. Shows the result and offers a redo,
    and warns about any flex channel that barely moved (unplugged / misplaced sensor)."""
    names = cfg["channels"]["analog"]
    n_a = len(names)
    while True:
        print("\n== Calibration ==")
        input("Press Enter, then hold your hand OPEN and flat.")
        simple_countdown("Open hand")
        seg_open = rec.start("_calib"); time.sleep(3.0); rec.stop()
        input("Press Enter, then make a tight FIST.")
        simple_countdown("Fist")
        seg_fist = rec.start("_calib"); time.sleep(3.0); rec.stop()

        rows = rec.frames(seg_open) + rec.frames(seg_fist)
        rec.discard([seg_open, seg_fist])  # calibration frames aren't training data
        if len(rows) < cfg["sample_rate_hz"]:
            sys.exit(f"Only {len(rows)} frames during calibration; check the port and that the "
                     f"firmware has DATA_COLLECTION_MODE 1.")
        calib = calib_from_frames(np.array([r[3:3 + n_a] for r in rows], dtype=float), cfg)

        fingers = cfg.get("finger_names", {})
        print("\n  channel  finger      min     max    span")
        weak = []
        for name, lo, hi in zip(names, calib["min"], calib["max"]):
            span = hi - lo
            finger = fingers.get(name, "?")
            flag = "   <-- barely moved: sensor connected?" if span < MIN_CALIB_SPAN else ""
            if flag:
                weak.append(f"{name} ({finger})")
            print(f"  {name:>6}   {finger:<8} {lo:7.0f} {hi:7.0f} {span:7.0f}{flag}")
        if weak:
            print(f"\n  Warning: {', '.join(weak)} changed very little between open hand and fist.")
        if input("\nEnter = accept, r + Enter = redo calibration: ").strip().lower() != "r":
            return calib


def simple_countdown(msg, secs=2):
    for i in range(secs, 0, -1):
        print(f"\r  {msg} in {i}...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {msg} -- GO!        ", flush=True)


class Quit(Exception):
    pass


def record_signs(rec, keys, cfg, plan, args):
    """The main loop. `queue` holds the letters still to record; `history` holds the
    kept signs (newest last) so undo can pop them off in reverse order."""
    queue = deque(plan)
    history = []                  # [(letter, [sign_segment, relax_segment]), ...]
    redone = Counter()
    total = len(plan)
    CONTROL = {"u", " ", "q"}

    def undo_last():
        if not history:
            print("  (nothing to undo yet)")
            return
        letter, segs = history.pop()
        rec.discard(segs)
        queue.appendleft(letter)  # redo it next
        redone[letter] += 1
        print(f"  UNDO: removed the last '{letter}'. You'll redo it next.")

    def pause():
        print("\n  PAUSED. space = resume, u = undo last sign, q = quit and save")
        while True:
            k = keys.wait_forever(CONTROL)
            if k == " ":
                print("  Resuming...")
                return
            if k == "u":
                undo_last()
            if k == "q":
                raise Quit

    def handle(key):
        """React to a control key pressed between signs."""
        if key == "u":
            undo_last()
        elif key == " ":
            pause()
        elif key == "q":
            raise Quit

    try:
        while queue:
            letter = queue[0]
            print(f"\n[{len(history) + 1}/{total}] Next sign: {letter}")

            # --- get ready --------------------------------------------------
            if args.self_paced:
                print("  Press Enter when ready.  (u = undo, space = pause, q = quit)")
                key = keys.wait_forever(CONTROL | {"\n"})
                if key != "\n":
                    handle(key)
                    continue
            key = None
            for i in range(2, 0, -1):
                print(f"\r  Sign {letter} in {i}...  ", end="", flush=True)
                key = keys.wait(1.0, CONTROL)
                if key:
                    break
            if key:
                print()
                handle(key)
                continue      # whatever happened, re-read the front of the queue
            print(f"\r  Sign {letter} -- GO!        ", flush=True)

            # --- hold the sign ---------------------------------------------
            hold = args.hold + (0.5 if letter in cfg["motion_classes"] else 0)
            seg = rec.start(letter)
            key = keys.wait(hold, CONTROL)
            n = rec.stop()
            if key:
                # Interrupted mid-sign: that attempt is incomplete, never keep it.
                rec.discard([seg])
                if key == "q":
                    raise Quit
                redone[letter] += 1
                print(f"  Discarded this '{letter}' attempt. Redoing it.")
                if key == " ":
                    pause()
                continue

            # --- relax (recorded as 'none') ---------------------------------
            print(f"  RELAX  ({n} frames)")
            rest_seg = rec.start("none")
            key = keys.wait(args.rest, CONTROL)
            rec.stop()
            queue.popleft()
            history.append((letter, [seg, rest_seg]))
            if key:
                handle(key)
    except (Quit, KeyboardInterrupt):
        print("\n\nStopped early; saving everything kept so far.")
    return history, redone


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, or 'fake' for a synthetic stream")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wearer")
    ap.add_argument("--session")
    ap.add_argument("--letters", nargs="*", help="subset of letters (default: all in config.yaml)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds per static sign (+0.5 for J/Z)")
    ap.add_argument("--rest", type=float, default=1.5, help="seconds of 'none' after each sign")
    ap.add_argument("--self-paced", action="store_true", help="press Enter to start each sign")
    ap.add_argument("--list-ports", action="store_true")
    args = ap.parse_args()

    if args.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        return
    if not (args.port and args.wearer and args.session):
        ap.error("--port, --wearer and --session are required")

    cfg = load_config()
    out = ROOT / "data" / "raw" / args.wearer / args.session
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists; pick a new --session name so nothing is overwritten.")
    letters = args.letters or [c for c in cfg["classes"] if c != "none"]
    unknown = set(letters) - set(cfg["classes"])
    if unknown:
        sys.exit(f"Letters not in config.yaml classes: {sorted(unknown)}")

    src = FakeSource(cfg) if args.port == "fake" else SerialSource(args.port, args.baud)
    rec = Recorder(src, cfg)
    time.sleep(1.0)
    if rec.good_lines == 0:
        print(f"Warning: no valid data lines yet ({rec.bad_lines} unreadable). "
              f"Is the firmware built with DATA_COLLECTION_MODE 1?")

    print("Put the glove on.")
    calib = calibrate(rec, cfg)

    plan = [c for c in letters for _ in range(args.reps)]
    random.shuffle(plan)
    pace = "press Enter before each sign" if args.self_paced else f"{args.hold:g} s per sign"
    print(f"\n== Recording {len(plan)} signs ({len(letters)} letters x {args.reps}, {pace}) ==")
    print("Hold each sign until it says RELAX.")
    print("Keys: u = undo last sign   space = pause   q = quit and save")
    with Keys() as keys:
        history, redone = record_signs(rec, keys, cfg, plan, args)
    rec.running = False

    # --- save ------------------------------------------------------------------
    cols = ["t_ms", "label", "segment"] + cfg["csv_columns"]
    df = pd.DataFrame(rec.rows, columns=cols)
    for c in cfg["channels"]["analog"] + cfg["channels"]["digital"]:
        df[c] = df[c].astype(int)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "recording.csv", index=False)
    (out / "calib.json").write_text(json.dumps(calib, indent=2))
    kept = Counter(letter for letter, _ in history)
    (out / "meta.json").write_text(json.dumps({
        "wearer": args.wearer, "session": args.session,
        "recorded": datetime.now().isoformat(timespec="seconds"),
        "port": args.port, "letters": letters, "reps_planned": args.reps,
        "hold_s": args.hold, "rest_s": args.rest, "self_paced": args.self_paced,
        "kept": dict(kept), "redone": dict(redone), "frames": len(df), "bad_lines": rec.bad_lines,
        "config": {k: cfg[k] for k in ("sample_rate_hz", "adc_bits", "csv_columns")},
    }, indent=2))

    # --- summary ---------------------------------------------------------------
    print(f"\nSaved {len(df)} frames to {out}")
    print(f"Signs kept: {sum(kept.values())}/{len(plan)}")
    short = {l: kept[l] for l in letters if kept[l] < args.reps}
    if short:
        print("Fewer reps than planned:", ", ".join(f"{l} {n}/{args.reps}" for l, n in short.items()))
    if redone:
        print("Redone (undo or interrupted):", ", ".join(f"{l} x{n}" for l, n in redone.most_common()))
        print("  Letters redone often may be hard to sign consistently; watch them in the confusion matrix.")


if __name__ == "__main__":
    main()
