"""
preprocess_landmarks.py
────────────────────────────────────────────────────────────────
Extracts MediaPipe hand landmarks from every video clip and saves
them as .npy files for LSTM training.

Output per clip: (SEQ_LEN, 126)
    First  63 values : normalised 3-D landmark positions
    Next   63 values : frame-to-frame velocity (delta)

Improvements over v1:
  - Handedness normalization  : left-hand x-coords are mirrored to
    right-hand convention so both hands map to the same space.
  - Velocity features         : frame-to-frame deltas encode HOW
    hands move, not just where they are — critical for signs that
    look similar in static frames but differ in motion.
────────────────────────────────────────────────────────────────
Usage:
    python scripts/preprocess_landmarks.py
"""

import sys
import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
OUT_DIR       = Path("data/landmarks")
LABELS_FILE   = Path("data/class_labels.json")
SEQ_LEN       = 30
LANDMARK_DIM  = 126    # 63 positions + 63 velocities
MIN_CLIPS     = 3

# ── MediaPipe (static mode — frames are sampled non-consecutively) ──
_mp_hands = mp.solutions.hands
HANDS     = _mp_hands.Hands(
    static_image_mode       = True,
    max_num_hands           = 1,
    min_detection_confidence= 0.4,
)


# ─────────────────────────────────────────────────────────────
#  LANDMARK EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_landmarks_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a (63,) float32 position vector from one frame.
    • Wrist-centred + scale-normalised → location/size invariant.
    • Left-hand x-coords mirrored → same spatial layout as right hand.
    Returns zeros if no hand detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result    = HANDS.process(frame_rgb)

    if result.multi_hand_landmarks:
        lm     = result.multi_hand_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        # Wrist-centre
        wrist  = coords[0].copy()
        coords = coords - wrist

        # Scale-normalise
        scale  = np.max(np.abs(coords)) + 1e-6
        coords = coords / scale

        # Mirror left hand → right-hand convention
        if result.multi_handedness:
            label = result.multi_handedness[0].classification[0].label
            if label == "Left":
                coords[:, 0] *= -1

        return coords.flatten()   # (63,)

    return np.zeros(63, dtype=np.float32)


def video_to_landmark_sequence(video_path: Path) -> np.ndarray:
    """
    Reads a video, samples SEQ_LEN frames evenly, extracts landmarks.
    Computes per-frame velocity (delta).
    Returns shape (SEQ_LEN, 126).
    """
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return np.zeros((SEQ_LEN, LANDMARK_DIM), dtype=np.float32)

    # Sample SEQ_LEN evenly
    indices  = np.linspace(0, len(frames) - 1, SEQ_LEN, dtype=int)
    sampled  = [frames[i] for i in indices]

    # Extract position landmarks for each sampled frame
    positions = np.stack(
        [extract_landmarks_from_frame(f) for f in sampled], axis=0
    )   # (SEQ_LEN, 63)

    # Velocity: diff between consecutive frames; first frame gets zero delta
    velocity = np.diff(positions, axis=0, prepend=positions[:1])  # (SEQ_LEN, 63)

    return np.concatenate([positions, velocity], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  SPLIT PROCESSING
# ─────────────────────────────────────────────────────────────

def process_split(clips: list, split_name: str) -> tuple:
    saved   = 0
    skipped = 0

    for video_path, class_name in tqdm(clips, desc=f"  [{split_name}]"):
        out_dir = OUT_DIR / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / (video_path.stem + ".npy")
        if out_path.exists():
            saved += 1
            continue

        seq = video_to_landmark_sequence(video_path)

        # Skip if position block is all zeros (no hand in any frame)
        if np.all(seq[:, :63] == 0):
            skipped += 1
            continue

        np.save(out_path, seq)
        saved += 1

    return saved, skipped


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ASL Landmark Extraction Pipeline")
    print(f"  Output dim per frame: {LANDMARK_DIM} (63 pos + 63 vel)")
    print("=" * 60)

    all_clips  = []
    label_map  = {}
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        clips = list(class_dir.glob("*.mp4"))
        if len(clips) < MIN_CLIPS:
            continue
        label_map[class_dir.name] = len(label_map)
        for clip in clips:
            all_clips.append((clip, class_dir.name))

    print(f"\n  Classes    : {len(label_map)}")
    print(f"  Total clips: {len(all_clips)}")

    if not all_clips:
        print("\n  No clips found in data/raw/. Run download_dataset.py first.")
        return

    paths  = [c[0] for c in all_clips]
    labels = [c[1] for c in all_clips]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    # No stratify on the small remainder — too few samples per class
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42
    )
    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}\n")

    splits = [
        (list(zip(X_tr,  y_tr)),  "train"),
        (list(zip(X_val, y_val)), "val"),
        (list(zip(X_te,  y_te)),  "test"),
    ]

    total_saved = total_skip = 0
    for clips, name in splits:
        s, sk = process_split(clips, name)
        total_saved += s
        total_skip  += sk

    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Landmark extraction complete!")
    print(f"  Saved  : {total_saved} sequences  shape ({SEQ_LEN}, {LANDMARK_DIM})")
    print(f"  Skipped: {total_skip} (no hand detected)")
    print(f"  Output : {OUT_DIR.resolve()}")
    print(f"  Next   → python model/lstm_train.py")
    print(f"{'='*60}")

    HANDS.close()


if __name__ == "__main__":
    main()
