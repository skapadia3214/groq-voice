# groq-voice

Realtime voice assistant powered by Groq's Llama, with **pluggable speech providers**: speech-to-text via Groq Whisper **or 60db**, and text-to-speech via ElevenLabs **or 60db** — switchable with a single environment variable.

## Features

- **Realtime Speech Recognition**: Uses Groq's Whisper API (or 60db) for accurate and fast speech-to-text conversion.
- **Intelligent Responses**: Powered by Groq's Llama to provide intelligent and context-aware responses.
- **Natural Sounding Speech**: Utilizes ElevenLabs (or 60db) advanced text-to-speech for natural and expressive audio output.
- **Pluggable Providers**: TTS and STT each sit behind a small provider abstraction, so you can switch between **ElevenLabs/Groq** and **60db** with a single environment variable — no code changes.

## Providers

Both speech-to-text and text-to-speech are selected at startup via environment variables:

| Variable | Options | Default | Required keys |
|----------|---------|---------|---------------|
| `TTS_PROVIDER` | `elevenlabs`, `60db` | `elevenlabs` | `ELEVENLABS_API_KEY` *or* `SIXTYDB_API_KEY` + `SIXTYDB_VOICE_ID` |
| `STT_PROVIDER` | `groq`, `60db` | `groq` | `GROQ_API_KEY` *or* `SIXTYDB_API_KEY` |

The Groq Llama agent always runs, so `GROQ_API_KEY` is always required. Only the keys for the providers you actually select are validated at startup.

To list your 60db voice IDs (for `SIXTYDB_VOICE_ID`):

```python
from providers import get_60db_voices
for v in get_60db_voices():
    print(v["voice_id"], v["name"])
```

## Installation

### Prerequisites

Ensure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/).

### Step-by-Step Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/groq-voice.git
   cd groq-voice
   ```

2. **Install Requirements**

   Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**

   Rename the `.env.example` file to `.env`:

   ```bash
   mv .env.example .env
   ```

   Open the `.env` file and fill in your API keys:

   - `GROQ_API_KEY` *(always required)*: Create an account and retrieve your API key from the [Groq Console](https://console.groq.com/keys).
   - `ELEVENLABS_API_KEY` *(required when `TTS_PROVIDER=elevenlabs`)*: Get your API key from your [ElevenLabs profile](https://elevenlabs.io/).
   - `SIXTYDB_API_KEY` *(required when `TTS_PROVIDER=60db` or `STT_PROVIDER=60db`)*: Get your API key from [60db](https://60db.ai/).
   - `SIXTYDB_VOICE_ID` *(required when `TTS_PROVIDER=60db`)*: A voice id from `GET /myvoices` (see the snippet above).

4. **Run the Program**

   Start the voice assistant:

   ```bash
   python voice_assistant.py
   ```

## Usage

Once the program is running, simply speak into your microphone. The assistant will recognize your speech, process it, and respond with a natural-sounding voice.

### Switching providers

Provider selection happens entirely in `.env` — no code changes:

```bash
# Default stack (unchanged): ElevenLabs TTS + Groq Whisper STT
TTS_PROVIDER=elevenlabs
STT_PROVIDER=groq

# Use 60db for voice output only:
TTS_PROVIDER=60db
SIXTYDB_API_KEY=sk_live_...
SIXTYDB_VOICE_ID=fbb75ed2-975a-40c7-9e06-38e30524a9a1

# Use 60db for everything (TTS + STT):
TTS_PROVIDER=60db
STT_PROVIDER=60db
```

## Architecture

```
mic ──▶ STTProvider ──▶ Groq Llama agent ──▶ TTSProvider ──▶ speakers
        (groq | 60db)    (langchain_groq)     (elevenlabs | 60db)
```

- **`providers.py`** — the abstraction layer. `TTSProvider.synthesize()` always returns an MP3 stream; `STTProvider.transcribe()` always returns text. Each interface has an ElevenLabs/Groq and a 60db implementation, chosen by the `build_*_provider()` factories.
- **`voice_assistant.py`** — capture, silence detection, playback, and the main loop. It talks only to the provider interfaces, so it is provider-agnostic.
- **`agent.py`** — LangChain + Groq Llama conversational agent with buffer memory.
- **`config.py`** — loads env vars and validates only the credentials the selected providers need.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and passes all tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue in the repository or contact us at skapadia@groq.com.

---