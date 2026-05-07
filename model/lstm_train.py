"""
lstm_train.py
────────────────────────────────────────────────────────────────
Trains the BiLSTM + Attention model on pre-extracted landmark
sequences (positions + velocities → 126 features per frame).

Usage:
    python model/lstm_train.py
    python model/lstm_train.py --classes 20 --epochs 150 --batch 32
────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lstm_architecture import build_lstm_model, compile_lstm_model

LANDMARKS_DIR = ROOT_DIR / "data" / "landmarks"
LABELS_FILE   = ROOT_DIR / "data" / "class_labels.json"
SAVE_DIR      = SCRIPT_DIR / "saved_lstm"
SAVE_DIR.mkdir(exist_ok=True)

SEQ_LEN      = 30
LANDMARK_DIM = 126    # 63 positions + 63 velocities


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_split(split: str, label_map: dict, num_classes: int):
    split_dir = LANDMARKS_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Landmarks not found at {split_dir}\n"
            "Run: python scripts/preprocess_landmarks.py"
        )

    X, y_raw = [], []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        word = class_dir.name
        if word not in label_map:
            continue
        idx = label_map[word]
        for npy_file in class_dir.glob("*.npy"):
            seq = np.load(npy_file)
            if seq.shape != (SEQ_LEN, LANDMARK_DIM):
                continue
            X.append(seq)
            y_raw.append(idx)

    if not X:
        raise ValueError(
            f"No {LANDMARK_DIM}-dim samples found for split '{split}'.\n"
            "Re-run: python scripts/preprocess_landmarks.py"
        )

    X = np.stack(X, axis=0).astype(np.float32)
    y = tf.keras.utils.to_categorical(y_raw, num_classes).astype(np.float32)
    print(f"  {split:5s}: {len(X):>5} samples  shape {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────────────────────

def _temporal_shift(X: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """
    Shift sequence by 1-max_shift frames, padding with the boundary
    frame instead of wrapping (no fake discontinuities).
    """
    out    = X.copy()
    shifts = np.random.randint(1, max_shift + 1, size=len(X))
    for i, s in enumerate(shifts):
        if np.random.rand() < 0.5:
            # shift right: pad start with first frame
            out[i] = np.concatenate([
                np.tile(X[i, :1], (s, 1)),
                X[i, :-s]
            ], axis=0)
        else:
            # shift left: pad end with last frame
            out[i] = np.concatenate([
                X[i, s:],
                np.tile(X[i, -1:], (s, 1))
            ], axis=0)
    return out


def _flip_hand(X: np.ndarray) -> np.ndarray:
    """
    Mirror x-coordinates to simulate the opposite hand signing.
    X-coords are at every 3rd position in both the position block
    (indices 0, 3, …, 60) and the velocity block (63, 66, …, 123).
    Using slice ::3 covers both.
    """
    out = X.copy()
    out[:, :, ::3] *= -1
    return out


def _speed_warp(X: np.ndarray) -> np.ndarray:
    """
    Resample sequence along time axis to simulate faster/slower signing.
    Speed factor drawn uniformly from [0.8, 1.2].
    """
    out     = X.copy()
    N, T, F = X.shape
    for i in range(N):
        factor     = np.random.uniform(0.8, 1.2)
        new_len    = max(2, int(T * factor))
        src_idx    = np.linspace(0, T - 1, new_len)
        dst_idx    = np.linspace(0, new_len - 1, T)
        for f in range(F):
            out[i, :, f] = np.interp(dst_idx, np.arange(new_len),
                                      np.interp(src_idx, np.arange(T),
                                                X[i, :, f]))
    return out


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    factor: int = 5) -> tuple:
    """
    Create `factor` augmented copies of the training set.
    Each copy applies a random combination of:
      temporal shift, scale jitter, gaussian noise, flip, speed warp.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(factor):
        Xc = X.copy()

        # Temporal shift (always)
        Xc = _temporal_shift(Xc)

        # Scale jitter
        scales = np.random.uniform(0.88, 1.12, size=(len(Xc), 1, 1))
        Xc    *= scales

        # Gaussian noise
        Xc    += np.random.normal(0, 0.008, Xc.shape).astype(np.float32)

        # Speed warp (50 % chance)
        if np.random.rand() < 0.5:
            Xc = _speed_warp(Xc)

        # Hand flip (40 % chance — simulates opposite-hand signer)
        if np.random.rand() < 0.4:
            Xc = _flip_hand(Xc)

        X_aug.append(Xc)
        y_aug.append(y)

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    idx   = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def make_dataset(X, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────

def build_callbacks(save_dir: Path) -> list:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(save_dir / f"training_log_{ts}.csv"),
        ),
    ]


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'='*60}")
    print(f"  ASL BiLSTM + Attention Training")
    print(f"  Features  : {LANDMARK_DIM} per frame (63 pos + 63 vel)")
    print(f"  Landmarks : {LANDMARKS_DIR}")
    print(f"  Batch     : {args.batch}")
    print(f"  Epochs    : {args.epochs}")
    print(f"{'='*60}\n")

    with open(LABELS_FILE) as f:
        label_map = json.load(f)

    sorted_labels = sorted(label_map.items(), key=lambda x: x[1])[:args.classes]
    label_map     = {w: i for i, (w, _) in enumerate(sorted_labels)}
    num_classes   = len(label_map)
    print(f"  Classes   : {num_classes}\n")

    print("[1/4] Loading landmark sequences …")
    X_tr, y_tr = load_split("train", label_map, num_classes)
    X_vl, y_vl = load_split("val",   label_map, num_classes)
    X_te, y_te = load_split("test",  label_map, num_classes)

    print(f"\n  Augmenting training data ({args.aug_factor}x) …")
    X_tr, y_tr = augment_dataset(X_tr, y_tr, factor=args.aug_factor)
    print(f"  After augmentation: {len(X_tr)} train samples")

    y_indices  = np.argmax(y_tr, axis=1)
    unique_cls = np.unique(y_indices)
    weights    = compute_class_weight("balanced",
                                      classes=unique_cls,
                                      y=y_indices)
    class_weights = dict(zip(unique_cls.tolist(), weights.tolist()))

    train_ds = make_dataset(X_tr, y_tr, args.batch, shuffle=True)
    val_ds   = make_dataset(X_vl, y_vl, args.batch, shuffle=False)
    test_ds  = make_dataset(X_te, y_te, args.batch, shuffle=False)

    print("\n[2/4] Building model …")
    model = build_lstm_model(num_classes=num_classes,
                             seq_len=SEQ_LEN,
                             landmark_dim=LANDMARK_DIM)
    compile_lstm_model(model, num_classes, learning_rate=1e-3)
    model.summary(line_length=70)

    print("\n[3/4] Training …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=build_callbacks(SAVE_DIR),
        verbose=1,
    )

    hist_path = SAVE_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()},
                  f, indent=2)

    print("\n[4/4] Evaluating on test set …")
    results                    = model.evaluate(test_ds, verbose=0)
    test_loss, test_acc, test_top5 = results[0], results[1], results[2]

    model.save(str(SAVE_DIR / "asl_lstm_model.keras"))

    with open(SAVE_DIR / "class_labels.json", "w") as f:
        json.dump(label_map, f, indent=2)

    meta = {
        "num_classes"   : num_classes,
        "seq_len"       : SEQ_LEN,
        "landmark_dim"  : LANDMARK_DIM,
        "best_val_acc"  : max(history.history.get("val_accuracy", [0])),
        "test_accuracy" : float(test_acc),
        "test_top5_acc" : float(test_top5),
    }
    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy  : {meta['best_val_acc']*100:.1f}%")
    print(f"  Test accuracy      : {test_acc*100:.1f}%")
    print(f"  Test top-5 accuracy: {test_top5*100:.1f}%")
    print(f"  Model → {SAVE_DIR / 'asl_lstm_model.keras'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL BiLSTM+Attn")
    parser.add_argument("--classes",    type=int, default=20)
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--aug-factor", type=int, default=5,
                        dest="aug_factor",
                        help="Training augmentation multiplier (default 5)")
    args = parser.parse_args()
    train(args)
