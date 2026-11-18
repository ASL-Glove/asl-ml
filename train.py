"""Train the letter classifier, then evaluate it on held-out sessions.

    python train.py --data data/synthetic    # fake data from synth.py (plumbing test)
    python train.py --data data/raw          # real recordings from collect.py

Where this fits:
    laptop:  recordings -> preprocess.py -> [THIS FILE: train float model] -> export.py -> int8 .tflite
    glove:   int8 .tflite as model_data.h -> LiteRT for Microcontrollers (TFLite Micro) runs it

Full TensorFlow is only used HERE, on the laptop, because LiteRT is an inference
runtime and cannot train. Whatever we build must use only layers LiteRT Micro
supports (Conv1D, pooling, Dense, ReLU, Softmax...) and must be small enough for the
Nano 33 BLE (~600 KB flash, ~190 KB RAM left).

Must produce, in artifacts/ (export.py depends on these):
    model.keras     trained float Keras model, input (window_frames, num_features)
    rep_data.npy    ~500 training windows, used by export.py to calibrate int8 quantization
Also written, for you to read:
    report.txt      accuracy, false-letter rate, top mix-ups, per-letter precision/recall
    confusion.png   27x27 grid of what was signed (rows) vs. what was predicted (columns)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from preprocess import ROOT, load_config, load_sessions, session_windows


def split_sessions(sessions, val_fraction, rng):
    """Hold out whole SESSIONS for validation, never random windows.
    Windows from the same session overlap and look nearly identical, so a random split
    would report fake-high accuracy. A held-out session behaves like putting the glove on again."""
    keys = sorted({(s.wearer, s.name) for s in sessions})
    if len(keys) < 2:
        raise SystemExit("Need at least 2 sessions so one can be held out for validation.")
    n_val = max(1, round(len(keys) * val_fraction))
    val = {keys[i] for i in rng.permutation(len(keys))[:n_val]}
    return ([s for s in sessions if (s.wearer, s.name) not in val],
            [s for s in sessions if (s.wearer, s.name) in val])


def stack(sessions, cfg):
    """All windows from a list of sessions -> X: (N, window_frames, num_features), y: (N,)"""
    parts = [session_windows(s, cfg) for s in sessions]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def build_model(window, n_features, n_classes):
    """Build the (untrained) neural network.

    Input : one window of sensor data, shape (window, n_features) = (50, 16)
            -> 50 time steps (1 s at 50 Hz), 16 sensor values per step.
    Output: n_classes (27) probabilities that add up to 1, one per class
            (index 0 = "none", 1 = A, ... 26 = Z). The highest one is the prediction.

    Design rules for the Arduino:
      * Only use layers LiteRT for Microcontrollers supports (all of these are).
      * Keep it small (~13k parameters -> ~24 KB as int8; we have ~600 KB flash free).
        With only a few recording sessions, a bigger model would just memorize them.
    """
    # Imported here (not at the top of the file) so `python train.py --help`
    # stays fast; TensorFlow takes a few seconds to load.
    import tensorflow as tf
    from tensorflow.keras import layers

    # A Sequential model is a straight pipeline: data flows through each layer
    # in order, top to bottom. Comments show the output shape after each layer
    # for one window (Keras adds a leading "batch" dimension we can ignore).
    return tf.keras.Sequential([
        # Declares what one input looks like. No learning happens here.
        layers.Input((window, n_features)),                          # -> (50, 16)

        # ---- Feature extraction: find patterns over TIME ----
        # Conv1D = a set of small "pattern detectors" (filters) that slide along
        # the 50 time steps. Each filter looks at `kernel_size` consecutive
        # readings across ALL 16 sensors at once, and learns to respond to one
        # pattern (e.g. "gyro X swings + then -", "finger 2 curled while D3 closed").
        #   24 filters  -> 24 different patterns learned
        #   kernel 5    -> each looks at 5 readings = 0.1 s of motion
        #   padding="same" keeps the length at 50 (pads the edges with zeros)
        #   relu        -> "only pass positive responses" (max(0, x)); this
        #                  non-linearity is what lets stacked layers learn complex shapes
        layers.Conv1D(24, kernel_size=5, padding="same", activation="relu"),  # -> (50, 24)

        # Keep the strongest response out of every 2 neighbouring steps.
        # Halves the time length: faster, smaller, and tolerant of a sign
        # happening a few milliseconds earlier or later.
        layers.MaxPooling1D(pool_size=2),                            # -> (25, 24)

        # Second detector layer. Its input is the FIRST layer's pattern scores,
        # so it learns combinations of patterns ("twist" + "pinky out" = J).
        # Because time was halved, its 5-step kernel now covers ~0.2 s.
        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),  # -> (25, 32)
        layers.MaxPooling1D(pool_size=2),                            # -> (12, 32)

        # Third detector layer: higher-level patterns spanning ~0.25-0.5 s,
        # long enough to capture the whole J hook or Z zig-zag.
        layers.Conv1D(48, kernel_size=3, padding="same", activation="relu"),  # -> (12, 48)

        # Collapse the time axis: average each of the 48 pattern scores over
        # the window -> "how strongly did each pattern appear in this second?"
        # After this the model no longer cares WHERE in the window the sign was.
        layers.GlobalAveragePooling1D(),                             # -> (48,)

        # ---- Classification: turn pattern scores into a letter ----
        # Training-only regularizer: randomly zeroes 20% of the values each step
        # so the model can't lean on any single pattern (reduces memorizing).
        # It's a no-op when the model runs on the glove.
        layers.Dropout(0.2),                                         # -> (48,)

        # Fully connected layer: every output looks at all 48 pattern scores
        # and learns which combinations go together.
        layers.Dense(32, activation="relu"),                         # -> (32,)

        # Final layer: one score per class. Softmax turns the 27 scores into
        # probabilities (all >= 0, sum = 1). The firmware uses the largest one,
        # and if it's below config.yaml's confidence_threshold it reports "none".
        layers.Dense(n_classes, activation="softmax"),               # -> (27,)
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(cfg["train"]["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.data, cfg)
    if not sessions:
        raise SystemExit(f"No sessions found under {args.data} (expected <wearer>/<session>/recording.csv)")
    train_s, val_s = split_sessions(sessions, cfg["train"]["val_fraction"], rng)
    print("train sessions:", [f"{s.wearer}/{s.name}" for s in train_s])
    print("val sessions:  ", [f"{s.wearer}/{s.name}" for s in val_s])

    Xtr, ytr = stack(train_s, cfg)
    Xva, yva = stack(val_s, cfg)
    print(f"windows: train {Xtr.shape}, val {Xva.shape}")
    print("class counts (train):", dict(zip(*np.unique(ytr, return_counts=True))))

    # Build the network and print its layers, output shapes, and parameter count.
    model = build_model(cfg["window_frames"], cfg["num_features"], len(cfg["classes"]))
    model.summary()

    # ======================= STEP 3: TRAINING =======================
    import tensorflow as tf

    # Same seed every run -> same random starting weights and shuffling,
    # so two runs on the same data give the same result (easier to compare changes).
    tf.keras.utils.set_random_seed(cfg["train"]["seed"])

    # --- compile(): tell the model HOW to learn -------------------------------
    # optimizer: the rule for adjusting weights after each batch. Adam is the
    #   standard choice: it adapts its step size automatically, so it works well
    #   without tuning. 0.001 (Keras's default) is how big each step is.
    # loss: the single number that measures "how wrong" the model is.
    #   "sparse categorical crossentropy" = for each window, -log(probability the
    #   model gave to the CORRECT class). Model says 90% K and it was K -> small
    #   loss (0.1). Says 5% K -> big loss (3.0). Training = pushing this down.
    #   "sparse" means labels are plain class numbers (0..26), not one-hot vectors.
    # metrics: extra numbers to print each epoch (accuracy = % of windows where
    #   the highest-probability class was the right one). Not used for learning.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- class weights: fix the "none" imbalance ------------------------------
    # The relax period after every sign is recorded as "none", so "none" has
    # ~13x more windows than any single letter. Left alone, the model would get
    # good accuracy just by saying "none" a lot. "Balanced" weights make each
    # mistake on a rare class count more: weight = total / (num_classes * count).
    # e.g. none: 4760 / (27 * 1560) ~= 0.11,  A: 4760 / (27 * 120) ~= 1.47
    classes_present, counts = np.unique(ytr, return_counts=True)
    class_weight = {int(c): len(ytr) / (len(classes_present) * n) for c, n in zip(classes_present, counts)}

    # --- early stopping: stop when it stops improving -------------------------
    # After every epoch we check the loss on the HELD-OUT session (val_loss).
    # If it hasn't improved for 8 epochs in a row, the model has started
    # memorizing the training sessions instead of learning signs, so we stop
    # and roll back to the weights from the best epoch.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)

    # --- fit(): the actual training loop -------------------------------------
    # Each EPOCH = one full pass over all training windows.
    # Within an epoch the windows are shuffled and fed in BATCHES of 64:
    #   1. model predicts the 64 windows
    #   2. loss measures how wrong it was
    #   3. backpropagation works out how each of the 12,931 weights contributed
    #   4. Adam nudges every weight slightly in the direction that lowers the loss
    # 4760 windows / 64 = ~75 weight updates per epoch.
    # validation_data is only EVALUATED (never learned from) at the end of each epoch.
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=cfg["train"]["epochs"],          # upper limit (40); early stopping usually ends sooner
        batch_size=cfg["train"]["batch_size"],  # 64
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,                              # one line per epoch
    )
    best = int(np.argmin(history.history["val_loss"]))
    print(f"best epoch: {best + 1}  val_loss {history.history['val_loss'][best]:.3f}  "
          f"val_accuracy {history.history['val_accuracy'][best]:.3f}")

    # --- save what export.py needs -------------------------------------------
    # model.keras: the trained float32 model (architecture + learned weights).
    model.save(out / "model.keras")
    # rep_data.npy: a random sample of 500 real training windows. export.py runs
    # these through the model to see the typical range of values at every layer,
    # which it needs to choose good int8 scales (the "representative dataset").
    np.save(out / "rep_data.npy", Xtr[rng.permutation(len(Xtr))[:500]])
    print(f"saved {out / 'model.keras'} and {out / 'rep_data.npy'}")

    # ======================= STEP 4: EVALUATION =======================
    # Everything here uses ONLY the held-out session(s): data the model never
    # trained on. That makes these numbers an honest preview of how the glove
    # will do the next time someone puts it on.
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    names = cfg["classes"]                  # index -> name: 0 "none", 1 "A", ... 26 "Z"
    all_idx = list(range(len(names)))

    # model.predict() returns, for every window, 27 probabilities (shape (N, 27)).
    # argmax(axis=1) picks the index of the highest one = the predicted class.
    probs = model.predict(Xva, verbose=0)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)                # how sure it was about that pick (0..1)

    # --- headline numbers ----------------------------------------------------
    # Overall accuracy is dominated by "none" (it's most of the windows), so we
    # also report accuracy on just the windows where a letter was being signed.
    acc = float((pred == yva).mean())
    is_letter = yva != 0
    letter_acc = float((pred[is_letter] == yva[is_letter]).mean()) if is_letter.any() else float("nan")

    # What the glove would actually do: config.yaml's confidence_threshold means
    # any prediction below 70% sure is reported as "none" instead of a letter.
    # Fewer wrong letters sent to the phone, at the cost of some missed ones.
    thr = cfg["confidence_threshold"]
    pred_thr = np.where(conf >= thr, pred, 0)
    acc_thr = float((pred_thr == yva).mean())
    # "False letters": the user wasn't signing (true = none) but a letter came out.
    # These are the most annoying errors in practice (random letters appearing).
    false_letter_rate = float((pred_thr[~is_letter] != 0).mean()) if (~is_letter).any() else float("nan")

    # --- per-class report ----------------------------------------------------
    # precision: when the model SAYS "K", how often is it really K?
    #            (low precision = other letters get mistaken FOR K)
    # recall:    when the user REALLY signs K, how often does the model say K?
    #            (low recall = K gets mistaken for something else)
    # f1:        a single score combining the two.
    # support:   how many held-out windows of that class there were.
    report = classification_report(yva, pred, labels=all_idx, target_names=names, zero_division=0)

    # --- confusion matrix ----------------------------------------------------
    # 27x27 grid. Row = what was really signed, column = what the model said.
    # A perfect model only has numbers on the diagonal. Anything off the
    # diagonal is a specific mix-up, e.g. row K / column P = "K read as P".
    cm = confusion_matrix(yva, pred, labels=all_idx)

    # List the worst mix-ups so you don't have to squint at the grid.
    # These point at what to fix: re-record those letters, check a sensor,
    # or accept that two letters look alike to this glove.
    off_diag = [(cm[i, j], names[i], names[j]) for i in all_idx for j in all_idx if i != j and cm[i, j] > 0]
    off_diag.sort(reverse=True)
    top_confusions = "\n".join(f"  {t:>4} -> {p:<4} {n} windows" for n, t, p in off_diag[:10]) or "  (none)"

    summary = (
        f"held-out sessions: {[f'{s.wearer}/{s.name}' for s in val_s]}\n"
        f"overall accuracy:            {acc:.3f}\n"
        f"letter-only accuracy:        {letter_acc:.3f}   (windows where a letter was signed)\n"
        f"accuracy w/ {thr:.0%} threshold:   {acc_thr:.3f}   (what the glove would do)\n"
        f"false-letter rate:           {false_letter_rate:.3f}   (resting hand -> letter sent anyway)\n\n"
        f"most common mix-ups (true -> predicted):\n{top_confusions}\n\n"
        f"{report}"
    )
    print("\n" + summary)
    (out / "report.txt").write_text(summary)

    # Save the grid as an image. Darker squares = more windows.
    import matplotlib
    matplotlib.use("Agg")                   # draw to a file, no window pops up
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actually signed")
    ax.set_title(f"Held-out session(s): accuracy {acc:.2f}, letters only {letter_acc:.2f}")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=110)
    print(f"saved {out / 'report.txt'} and {out / 'confusion.png'}")


if __name__ == "__main__":
    main()
