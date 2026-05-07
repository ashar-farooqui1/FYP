"""
inference.py  —  Real-time ASL inference engine
────────────────────────────────────────────────────────────────
Key design decisions for accurate real-time detection:

1. Buffer only fills when a HAND IS PRESENT.
   No-hand frames are NOT added to the buffer, so the model always
   sees a clean sequence of actual sign frames, not idle camera.

2. Buffer resets when hand disappears for NO_HAND_RESET_FRAMES.
   Prevents the previous sign from bleeding into the next one.

3. Velocity is only computed between consecutive hand-present frames.
   Avoids the large velocity spike that occurs when a hand suddenly
   appears or disappears from frame.

4. Prediction requires CONFIRM_STREAK identical results in a row.
   One-off wrong predictions are suppressed.

5. CONF_THRESHOLD filters low-confidence outputs.
────────────────────────────────────────────────────────────────
"""

import cv2
import json
import base64
import numpy as np
import mediapipe as mp
import keras
from pathlib import Path
from collections import deque, Counter

# ── Config ────────────────────────────────────────────────────
SEQ_LEN              = 30
LANDMARK_DIM         = 126    # 63 positions + 63 velocities
CONF_THRESHOLD       = 0.45
SMOOTH_WINDOW        = 7
CONFIRM_STREAK       = 3
NO_HAND_RESET_FRAMES = 8      # frames without hand before buffer clears

# ── MediaPipe ─────────────────────────────────────────────────
_mp_hands = mp.solutions.hands
_hands    = _mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
)


class ASLInferenceEngine:

    def __init__(self, model_path: Path, labels_path: Path):
        print(f"[Inference] Loading model from {model_path} ...")
        self.model = keras.models.load_model(str(model_path))

        with open(labels_path) as f:
            label_map = json.load(f)
        self.idx_to_word = {v: k for k, v in label_map.items()}
        self.num_classes  = len(label_map)

        self._reset_state()
        print(f"[Inference] Ready — {self.num_classes} classes")

    # ── Public API ────────────────────────────────────────────

    def process_frame(self, frame_b64: str) -> dict:
        frame_bgr        = self._decode_frame(frame_b64)
        pos, detected    = self._extract_position(frame_bgr)

        if detected:
            # Compute velocity only between consecutive hand frames
            vel = (pos - self._prev_pos) if self._prev_pos is not None \
                  else np.zeros(63, dtype=np.float32)
            self._prev_pos       = pos.copy()
            self._no_hand_streak = 0

            feature = np.concatenate([pos, vel])
            self.frame_buffer.append(feature)
        else:
            self._no_hand_streak += 1
            self._prev_pos = None   # reset velocity on hand loss

            if self._no_hand_streak >= NO_HAND_RESET_FRAMES:
                self._reset_state()

        buffer_fill = len(self.frame_buffer) / SEQ_LEN

        result = {
            "prediction"   : "",
            "confidence"   : 0.0,
            "top3"         : [],
            "buffer_fill"  : buffer_fill,
            "hand_detected": bool(detected),
        }

        if len(self.frame_buffer) == SEQ_LEN:
            pred, conf, top3 = self._predict()
            result.update({"prediction": pred, "confidence": conf, "top3": top3})

        return result

    def reset(self):
        self._reset_state()

    # ── Internal ──────────────────────────────────────────────

    def _reset_state(self):
        self.frame_buffer    = deque(maxlen=SEQ_LEN)
        self.pred_buffer     = deque(maxlen=SMOOTH_WINDOW)
        self._prev_pos       = None
        self._no_hand_streak = 0
        self._streak_word    = None
        self._streak_count   = 0

    def _decode_frame(self, frame_b64: str) -> np.ndarray:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        return cv2.imdecode(
            np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    def _extract_position(self, frame_bgr: np.ndarray) -> tuple:
        """
        Returns (position_vec: (63,), detected: bool).
        Matches preprocess_landmarks.py normalization exactly.
        """
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = _hands.process(rgb)

        if result.multi_hand_landmarks:
            lm     = result.multi_hand_landmarks[0].landmark
            coords = np.array([[p.x, p.y, p.z] for p in lm],
                               dtype=np.float32)
            wrist  = coords[0].copy()
            coords = coords - wrist
            scale  = np.max(np.abs(coords)) + 1e-6
            coords = coords / scale

            if result.multi_handedness:
                if result.multi_handedness[0].classification[0].label == "Left":
                    coords[:, 0] *= -1

            return coords.flatten(), True

        return np.zeros(63, dtype=np.float32), False

    def _predict(self) -> tuple:
        seq   = np.stack(self.frame_buffer, axis=0)[np.newaxis]  # (1,30,126)
        probs = self.model.predict(seq, verbose=0)[0]

        top_idx  = np.argsort(probs)[::-1]
        top3     = [[self.idx_to_word.get(int(i), "?"), float(probs[i])]
                    for i in top_idx[:3]]
        raw_word = self.idx_to_word.get(int(top_idx[0]), "?")
        raw_conf = float(probs[top_idx[0]])

        self.pred_buffer.append(raw_word)
        smooth_word = Counter(self.pred_buffer).most_common(1)[0][0]

        # Require same word N consecutive times before emitting
        if smooth_word == self._streak_word:
            self._streak_count += 1
        else:
            self._streak_word  = smooth_word
            self._streak_count = 1

        if raw_conf < CONF_THRESHOLD or self._streak_count < CONFIRM_STREAK:
            return "", raw_conf, top3

        return smooth_word, raw_conf, top3
