"""
realtime_demo.py
────────────────────────────────────────────────────────────────
Real-time ASL word recognition using webcam + trained BiLSTM.

Controls:
    Q  — quit
    C  — clear current sequence / reset
    S  — toggle smoothing
────────────────────────────────────────────────────────────────
Usage:
    python demo/realtime_demo.py
    python demo/realtime_demo.py --model model/saved_lstm/asl_lstm_model.keras
────────────────────────────────────────────────────────────────
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque, Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import mediapipe as mp

SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
DEFAULT_MODEL = ROOT_DIR / "model" / "saved_lstm" / "asl_lstm_model.keras"
DEFAULT_LABELS= ROOT_DIR / "model" / "saved_lstm" / "class_labels.json"

SEQ_LEN        = 30
LANDMARK_DIM   = 126   # 63 positions + 63 velocities
CONF_THRESHOLD = 0.40
SMOOTH_WINDOW  = 5

GREEN  = (0, 220, 100)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
YELLOW = (0, 215, 255)


# ─────────────────────────────────────────────────────────────
#  LANDMARK EXTRACTION  (matches preprocess_landmarks.py exactly)
# ─────────────────────────────────────────────────────────────

def extract_position(frame_bgr: np.ndarray, hands_detector) -> tuple:
    """
    Returns (position_vec: (63,), hand_landmarks_for_drawing | None).
    Applies wrist-centring, scale-normalisation, and handedness mirror.
    """
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if result.multi_hand_landmarks:
        lm     = result.multi_hand_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        wrist  = coords[0].copy()
        coords = coords - wrist
        scale  = np.max(np.abs(coords)) + 1e-6
        coords = coords / scale

        if result.multi_handedness:
            label = result.multi_handedness[0].classification[0].label
            if label == "Left":
                coords[:, 0] *= -1

        return coords.flatten(), result.multi_hand_landmarks[0]

    return np.zeros(63, dtype=np.float32), None


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(img, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(img, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(img, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(img, (x2 - r, y2 - r), r, color, thickness)


def draw_ui(frame, prediction, confidence, top3, buffer_fill, stabilize):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "ASL Translator", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    stab_txt = "Smooth: ON" if stabilize else "Smooth: OFF"
    cv2.putText(frame, stab_txt, (w - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    bar_w = int((w - 20) * buffer_fill)
    cv2.rectangle(frame, (10, 55), (w - 10, 70), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 55), (10 + bar_w, 70), GREEN, -1)
    cv2.putText(frame, "Collecting frames…", (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

    if prediction and confidence >= CONF_THRESHOLD:
        box_y = h - 180
        overlay2 = frame.copy()
        draw_rounded_rect(overlay2, 10, box_y, 340, h - 10, 12, BLACK, -1)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "Prediction", (20, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)
        cv2.putText(frame, prediction.upper(), (20, box_y + 75),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, GREEN, 3)

        conf_bar = int(300 * confidence)
        cv2.rectangle(frame, (20, box_y + 90), (320, box_y + 105), (60, 60, 60), -1)
        color = GREEN if confidence > 0.6 else YELLOW
        cv2.rectangle(frame, (20, box_y + 90), (20 + conf_bar, box_y + 105), color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (326, box_y + 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.putText(frame, "Top 3:", (20, box_y + 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        for i, (word, conf) in enumerate(top3):
            cv2.putText(frame, f"{i+1}. {word}  {conf*100:.0f}%",
                        (20, box_y + 148 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        WHITE if i == 0 else (140, 140, 140), 1)

    cv2.putText(frame, "Q: quit  C: clear  S: smooth",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    return frame


# ─────────────────────────────────────────────────────────────
#  MAIN DEMO LOOP
# ─────────────────────────────────────────────────────────────

def run_demo(model_path: Path, labels_path: Path):
    print(f"Loading model from {model_path} …")
    model = tf.keras.models.load_model(str(model_path))

    with open(labels_path) as f:
        label_map = json.load(f)
    idx_to_word = {v: k for k, v in label_map.items()}
    print(f"Loaded {len(label_map)} classes. Press Q to quit.\n")

    mp_hands_mod = mp.solutions.hands
    mp_draw      = mp.solutions.drawing_utils
    hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_buffer = deque(maxlen=SEQ_LEN)   # each entry: (126,)
    pred_buffer  = deque(maxlen=SMOOTH_WINDOW)
    prev_pos     = None    # for velocity computation
    prediction   = ""
    confidence   = 0.0
    top3         = []
    stabilize    = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        pos, hand_lms = extract_position(frame, hands)

        # Velocity = delta from previous frame's position
        vel      = pos - prev_pos if prev_pos is not None else np.zeros(63, dtype=np.float32)
        prev_pos = pos.copy()

        feature = np.concatenate([pos, vel])   # (126,)
        frame_buffer.append(feature)

        if hand_lms:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=GREEN, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=WHITE, thickness=1),
            )

        buffer_fill = len(frame_buffer) / SEQ_LEN
        if len(frame_buffer) == SEQ_LEN:
            seq   = np.stack(frame_buffer, axis=0)[np.newaxis]  # (1, 30, 126)
            probs = model.predict(seq, verbose=0)[0]

            top_idx  = np.argsort(probs)[::-1]
            top3     = [(idx_to_word.get(i, "?"), float(probs[i]))
                        for i in top_idx[:3]]
            raw_pred = idx_to_word.get(int(top_idx[0]), "?")
            raw_conf = float(probs[top_idx[0]])

            if stabilize:
                pred_buffer.append(raw_pred)
                prediction = Counter(pred_buffer).most_common(1)[0][0]
                confidence = raw_conf
            else:
                prediction = raw_pred
                confidence = raw_conf

        frame = draw_ui(frame, prediction, confidence, top3, buffer_fill, stabilize)
        cv2.imshow("ASL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            frame_buffer.clear()
            pred_buffer.clear()
            prev_pos   = None
            prediction = ""
            confidence = 0.0
            top3       = []
        elif key == ord("s"):
            stabilize = not stabilize

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Demo closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Real-time Demo")
    parser.add_argument("--model",  type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    model_path  = Path(args.model)
    labels_path = Path(args.labels)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python model/lstm_train.py first")
        sys.exit(1)
    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        sys.exit(1)

    run_demo(model_path, labels_path)
