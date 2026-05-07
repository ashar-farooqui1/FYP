"""
evaluate.py
────────────────────────────────────────────────────────────────
Evaluates the trained BiLSTM model on the test set and produces:
  - Accuracy, Top-5 Accuracy, Per-class F1
  - Confusion matrix (saved as PNG)
  - Classification report (saved as JSON)
────────────────────────────────────────────────────────────────
Usage:
    python model/evaluate.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              top_k_accuracy_score)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
LANDMARKS_DIR = ROOT_DIR / "data" / "landmarks"
SAVE_DIR      = SCRIPT_DIR / "saved_lstm"
MODEL_PATH    = SAVE_DIR / "asl_lstm_model.keras"
LABELS_PATH   = SAVE_DIR / "class_labels.json"
REPORT_DIR    = SAVE_DIR / "evaluation"
REPORT_DIR.mkdir(exist_ok=True)

SEQ_LEN      = 30
LANDMARK_DIM = 126   # 63 positions + 63 velocities


def load_test_data(label_map: dict, num_classes: int):
    test_dir = LANDMARKS_DIR / "test"
    X, y_raw = [], []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        word = class_dir.name
        if word not in label_map:
            continue
        for npy_file in class_dir.glob("*.npy"):
            seq = np.load(npy_file)
            if seq.shape != (SEQ_LEN, LANDMARK_DIM):
                continue
            X.append(seq)
            y_raw.append(label_map[word])

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y_raw, dtype=np.int32)
    return X, y


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    fig_size = max(12, len(class_names) // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix — ASL BiLSTM", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {out_path}")


def main():
    print(f"\n{'='*60}")
    print(f"  ASL BiLSTM — Evaluation")
    print(f"{'='*60}\n")

    # ── Load model & labels ───────────────────────────────────
    print("[1/4] Loading model …")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    with open(LABELS_PATH) as f:
        label_map = json.load(f)

    idx_to_word  = {v: k for k, v in label_map.items()}
    num_classes  = len(label_map)
    class_names  = [idx_to_word[i] for i in range(num_classes)]

    print(f"  Model    : {MODEL_PATH}")
    print(f"  Classes  : {num_classes}")

    # ── Load test data ────────────────────────────────────────
    print("\n[2/4] Loading test data …")
    X_te, y_te = load_test_data(label_map, num_classes)
    print(f"  Test samples: {len(X_te)}")

    # ── Run predictions ───────────────────────────────────────
    print("\n[3/4] Running predictions …")
    probs  = model.predict(X_te, verbose=0)       # (N, num_classes)
    y_pred = np.argmax(probs, axis=1)

    # ── Metrics ───────────────────────────────────────────────
    print("\n[4/4] Computing metrics …")

    acc      = np.mean(y_pred == y_te)
    top5_acc = top_k_accuracy_score(y_te, probs, k=min(5, num_classes))

    print(f"\n  Top-1 Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Top-5 Accuracy : {top5_acc:.4f}  ({top5_acc*100:.1f}%)")

    # Per-class report
    report = classification_report(
        y_te, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_te, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\n{report_str}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    plot_confusion_matrix(cm, class_names)

    # ── Save report ───────────────────────────────────────────
    summary = {
        "top1_accuracy": float(acc),
        "top5_accuracy": float(top5_acc),
        "num_test_samples": int(len(X_te)),
        "num_classes": num_classes,
        "per_class": report,
    }
    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ Report saved → {report_path}")
    print(f"\n{'='*60}")
    print(f"  Evaluation complete!")
    print(f"  Top-1 : {acc*100:.1f}%  |  Top-5 : {top5_acc*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()