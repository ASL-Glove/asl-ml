# asl-ml

Training pipeline for the ASL glove's letter classifier.

The model is trained here on a laptop with TensorFlow. It's then converted to a small int8 model that runs on the Arduino Nano 33 BLE Sense Rev2 with **LiteRT for Microcontrollers** (TensorFlow Lite Micro).

```
glove (DATA_COLLECTION_MODE 1) --USB--> collect.py --> data/raw/
data/raw/ --> train.py --> artifacts/model.keras --> export.py --> model_data.h --> firmware
```

## Setup

Needs Python 3.10–3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Try it without the glove

```bash
.venv/bin/python synth.py       # fake sessions -> data/synthetic/
.venv/bin/python train.py --data data/synthetic
.venv/bin/python export.py      # -> artifacts/firmware/model_data.h
```

Accuracy on the fake data is meaningless. It only proves the pipeline works.

## Recording real data

1. In `glove-firmware.ino`, set `#define DATA_COLLECTION_MODE 1` and upload.
2. Find the port: `.venv/bin/python collect.py --list-ports`
3. Record a session:
   ```bash
   .venv/bin/python collect.py --port <port> --wearer <name> --session s1
   ```
   - The script first calibrates: you hold your hand open, then make a fist. Then it prompts each letter in random order.
   - Hold each sign until it says **RELAX**.
   - Add `--self-paced` to press Enter before each sign instead of following a timer.
4. Take the glove off, put it back on, and record `s2`, `s3`, and so on.
   - Training needs **at least 2 sessions**, because one is held out to test on.
   - More sessions and more wearers give a better model.

Keys while recording (no Enter needed):

| Key | Action |
|---|---|
| `u` | Undo the last sign and redo it. Press again to go further back. |
| `space` | Pause / resume |
| `q` | Quit and save |

Recordings go in `data/`, which is **not** tracked by git. Share recordings with the team outside the repo (for example, a shared drive).

## Training and exporting

```bash
.venv/bin/python train.py                  # uses data/raw/
.venv/bin/python export.py --calib data/raw/<wearer>/<session>/calib.json
```

- `train.py` writes `artifacts/report.txt` (accuracy and the most common mix-ups) and `artifacts/confusion.png`.
- `export.py` writes `artifacts/firmware/model_data.h` and `preprocess_config.h` for the firmware.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Settings shared by every script: sample rate, window length, channels, letters, finger mapping. Changing the sample rate, window, or channels means re-recording. |
| `collect.py` | Records labeled sessions from the glove. |
| `preprocess.py` | Turns raw readings into model inputs. **The firmware must use the same math.** |
| `train.py` | Defines, trains, and evaluates the model (a small 1D CNN). |
| `export.py` | Converts the trained model to int8 LiteRT and generates the C headers. |
| `synth.py` | Generates fake sessions for testing. |

## Inputs (16 per reading, 50 readings per second)

| Columns | Sensor | Scaled to |
|---|---|---|
| `a0 a1 a2 a3 a6` | Flex: pinky, ring, middle, index, thumb | 0–1, using that session's calibration |
| `d2`–`d6` | Contact pads | 0 or 1 |
| `ax ay az` | Accelerometer, g (hand orientation) | ÷ 4 |
| `gx gy gz` | Gyroscope, deg/s (hand rotation) | ÷ 2000 |
