"""
download_dataset.py
────────────────────────────────────────────────────────────────
Downloads the WLASL (Word-Level American Sign Language) dataset
from Kaggle using the kagglehub API, then organises the raw
video files into a clean directory tree.

Dataset chosen: risangbaskoro/wlasl-processed
  • ~2000 ASL word classes
  • Flat MP4 clips named by video ID (e.g. 00335.mp4)
  • Class mapping provided by WLASL_v0.3.json
────────────────────────────────────────────────────────────────
Usage:
    1. Place your kaggle.json in ~/.kaggle/  OR set env vars:
           KAGGLE_USERNAME=<your_username>
           KAGGLE_KEY=<your_api_key>
    2. python scripts/download_dataset.py
"""

import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
import kagglehub
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────
load_dotenv()

RAW_DIR      = Path("data/raw")
LABELS_FILE  = Path("data/class_labels.json")
MAX_CLASSES  = 20   # top-200 most-represented words
                     # increase to 2000 once pipeline is proven

# ── Kaggle credentials ────────────────────────────────────────
os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
os.environ.setdefault("KAGGLE_KEY",      os.getenv("KAGGLE_KEY",      ""))


def download_via_kagglehub() -> Path:
    print("[1/4] Downloading WLASL-processed dataset via kagglehub …")
    dataset_path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    print(f"      ✓ Cached at: {dataset_path}")
    return Path(dataset_path)


def build_video_id_map(source: Path):
    """
    Reads WLASL_v0.3.json and builds a mapping:
        video_id (str, zero-padded 5-digit) -> word (class label)

    WLASL JSON structure:
    [
      {
        "gloss": "book",
        "instances": [
          { "video_id": "00335", "split": "train", ... },
          ...
        ]
      },
      ...
    ]
    """
    json_path = source / "WLASL_v0.3.json"
    if not json_path.exists():
        candidates = list(source.rglob("WLASL_v0.3.json"))
        if not candidates:
            raise FileNotFoundError(f"WLASL_v0.3.json not found inside {source}")
        json_path = candidates[0]

    print(f"      Reading class map from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count instances per gloss to pick top MAX_CLASSES
    gloss_counts = {}
    for entry in data:
        gloss = entry["gloss"]
        gloss_counts[gloss] = len(entry.get("instances", []))

    top_glosses = sorted(gloss_counts, key=gloss_counts.get, reverse=True)[:MAX_CLASSES]
    top_set     = set(top_glosses)

    # Build video_id -> gloss mapping (only for top classes)
    vid_to_gloss = {}
    for entry in data:
        gloss = entry["gloss"]
        if gloss not in top_set:
            continue
        for inst in entry.get("instances", []):
            vid_id = str(inst["video_id"]).zfill(5)
            vid_to_gloss[vid_id] = gloss

    label_map = {word: idx for idx, word in enumerate(sorted(top_glosses))}

    print(f"      ✓ {len(top_glosses)} classes | {len(vid_to_gloss)} video mappings loaded")
    return vid_to_gloss, label_map


def organise_into_class_dirs(source: Path) -> dict:
    print("[2/4] Organising clips into class directories …")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Find the flat videos folder
    video_root = source / "videos"
    if not video_root.exists():
        candidates = list(source.rglob("videos"))
        if not candidates:
            raise FileNotFoundError(f"'videos' folder not found inside {source}")
        video_root = candidates[0]

    print(f"      Videos folder: {video_root}")

    vid_to_gloss, label_map = build_video_id_map(source)

    all_mp4s = list(video_root.glob("*.mp4"))
    print(f"      Total MP4 files found: {len(all_mp4s)}")

    copied  = 0
    skipped = 0
    for mp4 in tqdm(all_mp4s, desc="  Copying clips"):
        vid_id = mp4.stem.zfill(5)       # e.g. "00335"
        gloss  = vid_to_gloss.get(vid_id)
        if gloss is None:
            skipped += 1
            continue                      # not in top-N classes

        dst_class = RAW_DIR / gloss
        dst_class.mkdir(parents=True, exist_ok=True)
        dest = dst_class / mp4.name
        if not dest.exists():
            shutil.copy2(mp4, dest)
            copied += 1

    print(f"      ✓ {copied} clips copied | {skipped} skipped (not in top-{MAX_CLASSES})")
    return label_map


def save_label_map(label_map: dict) -> None:
    print("[3/4] Saving class label map …")
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"      ✓ Saved → {LABELS_FILE}  ({len(label_map)} classes)")


def print_summary() -> None:
    print("[4/4] Dataset summary:")
    class_dirs  = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    total_clips = sum(len(list(d.glob("*.mp4"))) for d in class_dirs)
    print(f"      Classes : {len(class_dirs)}")
    print(f"      Clips   : {total_clips}")
    print(f"      Location: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    cache_path = download_via_kagglehub()
    label_map  = organise_into_class_dirs(cache_path)
    save_label_map(label_map)
    print_summary()
    print("\n✅  Download complete. Next → run  scripts/preprocess.py")