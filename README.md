# groq-voice

Realtime voice assistant powered by Groq's Whisper API, Groq's Llama, and ElevenLabs text-to-speech.

## Features

- **Realtime Speech Recognition**: Uses Groq's Whisper API for accurate and fast speech-to-text conversion.
- **Intelligent Responses**: Powered by Groq's Llama to provide intelligent and context-aware responses.
- **Natural Sounding Speech**: Utilizes ElevenLabs' advanced text-to-speech technology for natural and expressive audio output.

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

   - `GROQ_API_KEY`: Create an account and retrieve your API key from the [Groq Console](https://console.groq.com/keys).
   - `ELEVENLABS_API_KEY`: Create an account and get your API key from your [ElevenLabs profile](https://elevenlabs.io/).

4. **Run the Program**

   Start the voice assistant:

   ```bash
   python voice_assistant.py
   ```

## Usage

Once the program is running, simply speak into your microphone. The assistant will recognize your speech, process it, and respond with a natural-sounding voice.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and passes all tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue in the repository or contact us at skapadia@groq.com.

---