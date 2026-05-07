"""
app.py
────────────────────────────────────────────────────────────────
FastAPI backend for the ASL Translator.

Endpoints:
    POST /predict        — send a base64 frame, get prediction
    POST /reset          — clear frame buffer
    GET  /tts/{word}     — get MP3 audio for a word
    GET  /classes        — list all supported ASL words
    GET  /health         — server health check
────────────────────────────────────────────────────────────────
Usage:
    cd asl-translator
    uvicorn backend.app:app --reload --port 8000
────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from backend.inference   import ASLInferenceEngine
from backend.tts_engine  import text_to_speech

# ── Model paths ───────────────────────────────────────────────
MODEL_PATH  = ROOT_DIR / "model" / "saved_lstm" / "asl_lstm_model.keras"
LABELS_PATH = ROOT_DIR / "model" / "saved_lstm" / "class_labels.json"

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="ASL Translator API",
    description="Real-time American Sign Language translation backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load inference engine once at startup ─────────────────────
engine: ASLInferenceEngine | None = None

@app.on_event("startup")
async def startup():
    global engine
    if not MODEL_PATH.exists():
        print(f"[WARNING] Model not found at {MODEL_PATH}")
        print("          Run: python model/lstm_train.py first")
        return
    engine = ASLInferenceEngine(MODEL_PATH, LABELS_PATH)


# ── Request / Response models ─────────────────────────────────

class FrameRequest(BaseModel):
    frame: str          # base64-encoded JPEG (data URI or raw)

class PredictionResponse(BaseModel):
    prediction  : str
    confidence  : float
    top3        : list
    buffer_fill : float
    hand_detected: bool


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status" : "ok",
        "model"  : "loaded" if engine else "not loaded",
    }


@app.get("/classes")
async def get_classes():
    if not engine:
        raise HTTPException(503, "Model not loaded")
    return {
        "classes": list(engine.idx_to_word.values()),
        "count"  : engine.num_classes,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: FrameRequest):
    if not engine:
        raise HTTPException(503, "Model not loaded. Run lstm_train.py first.")
    try:
        result = engine.process_frame(req.frame)
        return result
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")


@app.post("/reset")
async def reset():
    if engine:
        engine.reset()
    return {"status": "buffer cleared"}


@app.get("/sign/{word}")
def get_sign(word: str):
    """
    Returns 30 frames of MediaPipe hand landmarks (x, y) for a given ASL word.
    Reads one of the raw training MP4s and extracts evenly-spaced frames.
    Results are cached in memory after first request.
    """
    import cv2
    import mediapipe as mp

    word_lower = word.strip().lower()

    if word_lower in _sign_cache:
        return _sign_cache[word_lower]

    raw_dir = ROOT_DIR / "data" / "raw" / word_lower
    if not raw_dir.exists():
        raise HTTPException(404, f"No sign data for '{word}'")

    mp4s = sorted(raw_dir.glob("*.mp4"))
    if not mp4s:
        raise HTTPException(404, f"No videos found for '{word}'")

    def extract(video_path: Path, target: int = 30) -> list:
        cap = cv2.VideoCapture(str(video_path))
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(frame)
        cap.release()
        if not raw:
            return []

        n = len(raw)
        indices = [int(i * (n - 1) / max(target - 1, 1)) for i in range(target)]

        mp_hands = mp.solutions.hands
        result_frames = []
        last_lm = None
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            min_detection_confidence=0.3) as hands:
            for idx in indices:
                rgb = cv2.cvtColor(raw[idx], cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    last_lm = [[float(p.x), float(p.y)] for p in lm]
                if last_lm:
                    result_frames.append(last_lm)
        return result_frames

    frames = []
    for mp4 in mp4s[:3]:
        frames = extract(mp4)
        if len(frames) >= 5:
            break

    if not frames:
        raise HTTPException(500, f"Could not extract hand landmarks for '{word}'")

    result = {"word": word_lower, "frames": frames}
    _sign_cache[word_lower] = result
    return result


_sign_cache: dict = {}


@app.get("/tts/{word}")
async def tts(word: str):
    """
    Returns MP3 audio for the given word.
    Frontend can play this as fallback if Web Speech API is unavailable.
    """
    try:
        audio = text_to_speech(word)
        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        raise HTTPException(500, f"TTS error: {str(e)}")