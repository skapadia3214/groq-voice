import os
from dotenv import load_dotenv
import pyaudio
from enum import Enum

load_dotenv()

# --- API keys -------------------------------------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional, legacy STT path
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SIXTYDB_API_KEY = os.getenv("SIXTYDB_API_KEY")

# --- Provider selection (env-switchable) ----------------------------------
# TTS_PROVIDER: "elevenlabs" (default) | "60db"
# STT_PROVIDER: "groq" (default)       | "60db"
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "groq").lower()

# --- 60db settings --------------------------------------------------------
SIXTYDB_BASE_URL = os.getenv("SIXTYDB_BASE_URL", "https://api.60db.ai")
SIXTYDB_VOICE_ID = os.getenv("SIXTYDB_VOICE_ID")  # required when TTS_PROVIDER=60db
# REST /tts-synthesize output format. Keep "mp3" so the playback pipeline
# (pydub decode -> PCM) stays identical to the ElevenLabs path.
SIXTYDB_TTS_OUTPUT_FORMAT = os.getenv("SIXTYDB_TTS_OUTPUT_FORMAT", "mp3")
SIXTYDB_STT_LANGUAGE = os.getenv("SIXTYDB_STT_LANGUAGE", "auto")

# --- Model / format constants ---------------------------------------------
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_22050_32")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")


class Voices(Enum):
    """ElevenLabs voice IDs."""
    APHRODITE = "fQuiOHUGZu5WDKWT80Wz"
    ADAM = "pNInz6obpgDQGcFmaJgB"
    CJ_MURPH = "876MHA6EtWKaHTEGzjy5"


# Default ElevenLabs voice (string id, not the enum member).
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", Voices.ADAM.value)

# --- Audio capture / playback constants -----------------------------------
VOICE_ID = Voices.ADAM
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 200  # Adjust this threshold as needed
SILENCE_DURATION = 2  # Duration of silence to stop recording in seconds
PRE_SPEECH_BUFFER_DURATION = 0.5  # 500ms of audio to keep before speech detection


def validate_config():
    """
    Validate only the credentials required by the currently selected providers,
    so a 60db-only user is not forced to supply ElevenLabs/OpenAI keys (and
    vice-versa). The Groq LLM agent is always required.
    """
    missing = []

    # LLM agent always runs on Groq.
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY (required for the Llama agent)")

    if TTS_PROVIDER == "elevenlabs":
        if not ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY (TTS_PROVIDER=elevenlabs)")
    elif TTS_PROVIDER == "60db":
        if not SIXTYDB_API_KEY:
            missing.append("SIXTYDB_API_KEY (TTS_PROVIDER=60db)")
        if not SIXTYDB_VOICE_ID:
            missing.append("SIXTYDB_VOICE_ID (TTS_PROVIDER=60db)")
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER!r} (use 'elevenlabs' or '60db')")

    if STT_PROVIDER == "groq":
        if not GROQ_API_KEY:
            missing.append("GROQ_API_KEY (STT_PROVIDER=groq)")
    elif STT_PROVIDER == "60db":
        if not SIXTYDB_API_KEY:
            missing.append("SIXTYDB_API_KEY (STT_PROVIDER=60db)")
    else:
        raise ValueError(f"Unknown STT_PROVIDER: {STT_PROVIDER!r} (use 'groq' or '60db')")

    if missing:
        raise ValueError(
            "Missing required configuration:\n  - " + "\n  - ".join(missing)
        )


validate_config()
