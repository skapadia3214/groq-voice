import wave
from time import time, sleep
from typing import Optional, Generator, Any
import pyaudio
import numpy as np
from io import BytesIO
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from .agent import Agent
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
    Voices
)


class VoiceAssistant:
    def __init__(
        self,
        voice_id: Optional[str] = Voices.ADAM,
        **kwargs,
    ):
        self.audio = pyaudio.PyAudio()
        self.agent = Agent()
        self.voice_id = voice_id
        self.xi_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.oai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.g_client = Groq(api_key=GROQ_API_KEY)
        self.silence_threshold = kwargs.get("silence_threshold", SILENCE_THRESHOLD)
        self.silence_duration = kwargs.get("silence_duration", SILENCE_DURATION)
        self.pre_speech_buffer_duration = kwargs.get("pre_speech_buffer_duration", PRE_SPEECH_BUFFER_DURATION)

    def is_silence(self, data, silence_threshold: Optional[float | int] = None):
        """
        Detect if the provided audio data is silence.

        Args:
            data (bytes): Audio data.

        Returns:
            bool: True if the data is considered silence, False otherwise.
        """
        if not silence_threshold:
            silence_threshold = self.silence_threshold
        audio_data = np.frombuffer(data, dtype=np.int16)
        mean = np.mean(audio_data**2)
        rms = np.sqrt((mean if mean >= 0 else 0))
        print(f"{rms=}")
        return rms < silence_threshold

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

    def speech_to_text(self, audio_bytes: BytesIO):
        """
        Transcribe speech to text using Groq.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        # Convert BytesIO to a valid WAV file
        audio_bytes.seek(0)
        wav_bytes = BytesIO()
        with wave.open(wav_bytes, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wav_file.setframerate(RATE)
            wav_file.writeframes(audio_bytes.read())

        wav_bytes.seek(0)

        # Transcribe the WAV file
        transcription = self.g_client.audio.transcriptions.create(
            file=("temp.wav", wav_bytes.read()),
            model="whisper-large-v3",
            response_format='verbose_json',
            temperature=0.0
        )
        print(transcription)
        return transcription.text

    def speech_to_text_o(self, audio_bytes: BytesIO):
        """
        Transcribe speech to text using Groq.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        # Convert BytesIO to a valid WAV file
        audio_bytes.seek(0)
        wav_bytes = BytesIO()
        with wave.open(wav_bytes, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wav_file.setframerate(RATE)
            wav_file.writeframes(audio_bytes.read())

        wav_bytes.seek(0)

        start = time()
        audio_bytes.seek(0)
        transcription = self.oai_client.audio.transcriptions.create(
            file=("temp.wav", wav_bytes.read()),
            model="whisper-1",
        )
        end = time()
        print(transcription)
        return transcription.text

    def text_to_speech(self, text, voice_id: Optional[str] = None):
        """
        Convert text to speech and return an audio stream.

        Args:
            text (str): The text to convert to speech.

        Returns:
            BytesIO: The audio stream.
        """
        voice_id = voice_id or self.voice_id
        response = self.xi_client.text_to_speech.convert(
            voice_id=voice_id,
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

    def audio_stream_to_iterator(self, audio_stream, format='mp3') -> Generator[Any, Any, None]:
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

    def chat(
        self,
        query: str,
        **kwargs
    ) -> str:
        """
        Chat with an LLM/Agent/Anything you want.
        Override this method if you want to proccess responses differently.

        Args:
            query (str): Convert speech to text from microphone input
        
        Returns:
            str: String output to be spoken
        """
        start = time()
        response = self.agent.chat(query, **kwargs)
        end = time()
        print(f"Response: {response}\nResponse Time: {end - start}")
        return response

    def run(self):
        """
        Main function to run the voice assistant.
        """
        while True:
            # STT
            audio_bytes = self.listen_for_speech()
            text = self.speech_to_text(audio_bytes)

            # Agent
            response_text = self.chat(text)
            
            # TTS
            audio_stream = self.text_to_speech(response_text)
            audio_iterator = self.audio_stream_to_iterator(audio_stream)
            self.stream_audio(audio_iterator)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
