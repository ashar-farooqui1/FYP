"""
tts_engine.py
────────────────────────────────────────────────────────────────
Server-side Text-to-Speech using gTTS.
Used as fallback when browser Web Speech API is unavailable.
────────────────────────────────────────────────────────────────
"""

import io
import hashlib
from pathlib import Path
from gtts import gTTS

# Simple in-memory cache so same word isn't re-synthesized
_cache: dict[str, bytes] = {}


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Convert text to MP3 audio bytes.
    Returns cached result if available.
    """
    key = hashlib.md5(f"{text}_{lang}".encode()).hexdigest()

    if key in _cache:
        return _cache[key]

    tts    = gTTS(text=text, lang=lang, slow=False)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    audio_bytes = buffer.getvalue()

    _cache[key] = audio_bytes
    return audio_bytes


def clear_cache():
    _cache.clear()