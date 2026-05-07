"""
test_backend.py
────────────────────────────────────────────────────────────────
Integration tests for the FastAPI backend.

Endpoints tested:
  1. GET  /health   — server + model status
  2. GET  /classes  — list supported ASL words
  3. GET  /tts/{word} — returns MP3 audio bytes
  4. POST /predict  — base64 frame → prediction dict
  5. POST /reset    — clear frame buffer

Usage:
    # With backend running on :8000
    pytest backend/test_backend.py -v

    # Quick smoke test (no pytest required)
    python backend/test_backend.py
────────────────────────────────────────────────────────────────
"""

import cv2
import base64
import numpy as np
import requests

BASE_URL = "http://localhost:8000"


def _make_blank_frame_b64() -> str:
    """Encode a 480×640 black frame as base64 JPEG data URI."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data
    print(f"  health — model={data['model']}")


def test_classes():
    r = requests.get(f"{BASE_URL}/classes")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "classes" in data
    assert "count" in data
    assert data["count"] > 0
    print(f"  classes — {data['count']} words: {data['classes'][:5]}…")


def test_tts():
    r = requests.get(f"{BASE_URL}/tts/hello")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.headers["content-type"] == "audio/mpeg"
    assert len(r.content) > 0
    print(f"  tts — {len(r.content)} bytes for 'hello'")


def test_predict():
    """Send 31 blank frames so the buffer fills and a prediction is returned."""
    frame_b64 = _make_blank_frame_b64()

    # Reset first to start with a clean buffer
    requests.post(f"{BASE_URL}/reset")

    last_data = None
    for i in range(31):
        r = requests.post(
            f"{BASE_URL}/predict",
            json={"frame": frame_b64},
        )
        assert r.status_code == 200, f"Frame {i}: expected 200, got {r.status_code}"
        last_data = r.json()

    assert "prediction"   in last_data
    assert "confidence"   in last_data
    assert "top3"         in last_data
    assert "buffer_fill"  in last_data
    assert "hand_detected" in last_data
    assert last_data["buffer_fill"] == 1.0
    print(f"  predict — prediction='{last_data['prediction']}' "
          f"conf={last_data['confidence']:.3f} "
          f"hand={last_data['hand_detected']}")


def test_reset():
    r = requests.post(f"{BASE_URL}/reset")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "buffer cleared"
    print("  reset — buffer cleared")


if __name__ == "__main__":
    print("\nRunning backend smoke tests …")
    print("(make sure the server is running:  uvicorn backend.app:app --port 8000)\n")

    tests = [test_health, test_classes, test_tts, test_predict, test_reset]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except requests.exceptions.ConnectionError:
            print(f"\n  Cannot connect. Run: uvicorn backend.app:app --port 8000")
            break
        except AssertionError as e:
            print(f"\n  FAIL {t.__name__}: {e}")

    print(f"\n{'All' if passed == len(tests) else passed}/{len(tests)} tests passed.")
