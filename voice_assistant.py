import wave
import os
import pyaudio
import numpy as np
from io import BytesIO
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
from openai import OpenAI
from groq import Groq
from config import (
    ELEVENLABS_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    FORMAT,
    CHANNELS,
    RATE,
    CHUNK,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    PRE_SPEECH_BUFFER_DURATION,
)

class VoiceAssistant:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.xi_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.oai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.g_client = Groq(api_key=GROQ_API_KEY)

    def is_silence(self, data):
        """
        Detect if the provided audio data is silence.

        Args:
            data (bytes): Audio data.

        Returns:
            bool: True if the data is considered silence, False otherwise.
        """
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < SILENCE_THRESHOLD

    def listen_for_speech(self):
        """
        Continuously detect silence and start recording when speech is detected.
        
        Returns:
            BytesIO: The recorded audio bytes.
        """
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Listening for speech...")
        pre_speech_buffer = []
        pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

        while True:
            data = stream.read(CHUNK)
            pre_speech_buffer.append(data)
            if len(pre_speech_buffer) > pre_speech_chunks:
                pre_speech_buffer.pop(0)

            if not self.is_silence(data):
                print("Speech detected, start recording...")
                stream.stop_stream()
                stream.close()
                return self.record_audio(pre_speech_buffer)

    def record_audio(self, pre_speech_buffer):
        """
        Record audio until silence is detected.

        Args:
            pre_speech_buffer (list): Buffer containing pre-speech audio data.

        Returns:
            BytesIO: The recorded audio bytes.
        """
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = pre_speech_buffer.copy()

        silent_chunks = 0
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if self.is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                break

        stream.stop_stream()
        stream.close()

        audio_bytes = BytesIO()
        with wave.open(audio_bytes, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        audio_bytes.seek(0)

        return audio_bytes

    def speech_to_text(self, audio_bytes):
        """
        Transcribe speech to text using OpenAI.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        audio_bytes.seek(0)
        transcription = self.oai_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-1",
        )
        return transcription.text

    def speech_to_text_g(self, audio_bytes):
        """
        Transcribe speech to text using OpenAI.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        audio_bytes.seek(0)
        transcription = self.oai_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-1",
        )
        return transcription.text

    def text_to_speech(self, text):
        """
        Convert text to speech and return an audio stream.

        Args:
            text (str): The text to convert to speech.

        Returns:
            BytesIO: The audio stream.
        """
        response = self.xi_client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
        )

        audio_stream = BytesIO()

        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        audio_stream.seek(0)
        return audio_stream

    def audio_stream_to_iterator(self, audio_stream, format='mp3'):
        """
        Convert audio stream to an iterator of raw PCM audio bytes.

        Args:
            audio_stream (BytesIO): The audio stream.
            format (str): The format of the audio stream.

        Returns:
            bytes: The raw PCM audio bytes.
        """
        audio = AudioSegment.from_file(audio_stream, format=format)
        audio = audio.set_frame_rate(22050).set_channels(2).set_sample_width(2)  # Ensure the format matches pyaudio parameters
        raw_data = audio.raw_data

        chunk_size = 1024  # Adjust as necessary
        for i in range(0, len(raw_data), chunk_size):
            yield raw_data[i:i + chunk_size]

    def stream_audio(self, audio_bytes_iterator, rate=22050, channels=2, format=pyaudio.paInt16):
        """
        Stream audio in real-time.

        Args:
            audio_bytes_iterator (bytes): The raw PCM audio bytes.
            rate (int): The sample rate of the audio.
            channels (int): The number of audio channels.
            format (pyaudio format): The format of the audio.
        """
        stream = self.audio.open(format=format,
                                 channels=channels,
                                 rate=rate,
                                 output=True)

        try:
            for audio_chunk in audio_bytes_iterator:
                stream.write(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def run(self):
        """
        Main function to run the voice assistant.
        """
        while True:
            audio_bytes = self.listen_for_speech()
            text = self.speech_to_text_g(audio_bytes)
            print("Transcription:", text)
            response_text = "You said: " + text
            audio_stream = self.text_to_speech(response_text)
            audio_iterator = self.audio_stream_to_iterator(audio_stream)
            self.stream_audio(audio_iterator)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
