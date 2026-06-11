"""
Pluggable TTS / STT providers.

Every TTS provider returns an in-memory **MP3** stream (BytesIO seeked to 0),
and every STT provider returns a plain transcript string. Because the contract
is identical across providers, the rest of the app (voice_assistant.py) never
needs to know whether ElevenLabs or 60db is in use -- selection happens once,
via config, through the build_* factories at the bottom of this module.
"""
import base64
from abc import ABC, abstractmethod
from io import BytesIO
from time import time
from typing import List, Optional

import requests

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_TTS_MODEL,
    ELEVENLABS_VOICE_ID,
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    SIXTYDB_API_KEY,
    SIXTYDB_BASE_URL,
    SIXTYDB_STT_LANGUAGE,
    SIXTYDB_TTS_OUTPUT_FORMAT,
    SIXTYDB_VOICE_ID,
)


# ===========================================================================
# Text-to-Speech
# ===========================================================================
class TTSProvider(ABC):
    """Convert text into an in-memory MP3 stream (BytesIO, seeked to 0)."""

    #: container format the synthesize() output is in -- consumed by pydub.
    audio_format: str = "mp3"

    @abstractmethod
    def synthesize(self, text: str, voice_id: Optional[str] = None) -> BytesIO:
        ...


class ElevenLabsTTS(TTSProvider):
    audio_format = "mp3"

    def __init__(self, default_voice_id: str = ELEVENLABS_VOICE_ID):
        # Imported lazily so a 60db-only deploy needn't install/configure it.
        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.default_voice_id = default_voice_id

    def synthesize(self, text: str, voice_id: Optional[str] = None) -> BytesIO:
        voice_id = voice_id or self.default_voice_id
        response = self.client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format=ELEVENLABS_OUTPUT_FORMAT,
            text=text,
            model_id=ELEVENLABS_TTS_MODEL,
        )

        stream = BytesIO()
        for chunk in response:
            if chunk:
                stream.write(chunk)
        stream.seek(0)
        return stream


class SixtyDBTTS(TTSProvider):
    """60db REST text-to-speech via POST /tts-synthesize (base64 response)."""

    audio_format = SIXTYDB_TTS_OUTPUT_FORMAT

    def __init__(self, default_voice_id: Optional[str] = SIXTYDB_VOICE_ID):
        self.default_voice_id = default_voice_id

    def synthesize(self, text: str, voice_id: Optional[str] = None) -> BytesIO:
        voice_id = voice_id or self.default_voice_id
        payload = {
            "text": text,
            "output_format": SIXTYDB_TTS_OUTPUT_FORMAT,
        }
        if voice_id:
            payload["voice_id"] = voice_id

        resp = requests.post(
            f"{SIXTYDB_BASE_URL}/tts-synthesize",
            headers={
                "Authorization": f"Bearer {SIXTYDB_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(f"60db TTS failed: {data.get('message')}")

        audio = base64.b64decode(data["audio_base64"])
        stream = BytesIO(audio)
        stream.seek(0)
        return stream


# ===========================================================================
# Speech-to-Text
# ===========================================================================
class STTProvider(ABC):
    """Transcribe a WAV audio stream into text."""

    @abstractmethod
    def transcribe(self, audio_bytes: BytesIO) -> str:
        ...


class GroqWhisperSTT(STTProvider):
    def __init__(self):
        from groq import Groq

        self.client = Groq(api_key=GROQ_API_KEY)

    def transcribe(self, audio_bytes: BytesIO) -> str:
        start = time()
        audio_bytes.seek(0)
        transcription = self.client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model=GROQ_STT_MODEL,
        )
        print(f"[STT groq] {time() - start:.2f}s")
        return transcription.text


class SixtyDBSTT(STTProvider):
    """60db speech-to-text via POST /stt (multipart upload)."""

    def transcribe(self, audio_bytes: BytesIO) -> str:
        start = time()
        audio_bytes.seek(0)
        resp = requests.post(
            f"{SIXTYDB_BASE_URL}/stt",
            headers={"Authorization": f"Bearer {SIXTYDB_API_KEY}"},
            files={"file": ("temp.wav", audio_bytes.read(), "audio/wav")},
            data={"language": SIXTYDB_STT_LANGUAGE},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"[STT 60db] {time() - start:.2f}s lang={data.get('language')}")
        return data.get("text", "")


# ===========================================================================
# Voices
# ===========================================================================
def get_60db_voices() -> List[dict]:
    """Return the caller's 60db voices via GET /myvoices (the `data` array)."""
    resp = requests.get(
        f"{SIXTYDB_BASE_URL}/myvoices",
        headers={"Authorization": f"Bearer {SIXTYDB_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


# ===========================================================================
# Factories (provider selection happens here, driven by config)
# ===========================================================================
_TTS_ALIASES = {
    "elevenlabs": "elevenlabs", "11labs": "elevenlabs", "xi": "elevenlabs",
    "60db": "60db", "sixtydb": "60db",
}
_STT_ALIASES = {
    "groq": "groq", "whisper": "groq",
    "60db": "60db", "sixtydb": "60db",
}


def build_tts_provider(name: str) -> TTSProvider:
    key = _TTS_ALIASES.get((name or "").lower())
    if key == "elevenlabs":
        return ElevenLabsTTS()
    if key == "60db":
        return SixtyDBTTS()
    raise ValueError(f"Unknown TTS provider: {name!r} (use 'elevenlabs' or '60db')")


def build_stt_provider(name: str) -> STTProvider:
    key = _STT_ALIASES.get((name or "").lower())
    if key == "groq":
        return GroqWhisperSTT()
    if key == "60db":
        return SixtyDBSTT()
    raise ValueError(f"Unknown STT provider: {name!r} (use 'groq' or '60db')")
