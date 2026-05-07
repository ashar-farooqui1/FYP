"""
verify_data.py  —  Quick sanity check before training.
Run: python scripts/verify_data.py
"""
import json, numpy as np
from pathlib import Path
from collections import Counter

PROC_DIR    = Path("data/processed")
LABELS_FILE = Path("data/class_labels.json")

def verify():
    with open(LABELS_FILE) as f:
        label_map = json.load(f)

    print(f"{'SPLIT':<8} {'CLASSES':>8} {'SAMPLES':>9} {'SHAPE'}")
    print("-" * 45)
    for split in ["train", "val", "test"]:
        split_dir = PROC_DIR / split
        if not split_dir.exists():
            print(f"{split:<8}  NOT FOUND")
            continue
        files   = list(split_dir.rglob("*.npy"))
        classes = {f.parent.name for f in files}
        shape   = np.load(files[0]).shape if files else "—"
        print(f"{split:<8} {len(classes):>8} {len(files):>9}    {shape}")

    # Class balance (training set)
    train_dir = PROC_DIR / "train"
    if train_dir.exists():
        counts = Counter(f.parent.name
                         for f in train_dir.rglob("*.npy"))
        vals   = list(counts.values())
        print(f"\nClass balance (train):")
        print(f"  min={min(vals)}  max={max(vals)}  "
              f"mean={np.mean(vals):.1f}  std={np.std(vals):.1f}")

if __name__ == "__main__":
    verify()