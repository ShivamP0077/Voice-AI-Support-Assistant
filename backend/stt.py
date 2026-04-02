from groq import Groq

from backend.config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text using Groq Whisper API.

    Args:
        file_path: Path to the audio file (webm, mp3, wav, etc.)

    Returns:
        Transcribed text string.
    """
    with open(file_path, "rb") as audio_file:
        transcription = _client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
        )

    return transcription.text
