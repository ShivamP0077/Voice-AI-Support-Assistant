import edge_tts


async def synthesize_speech(text: str, output_path: str) -> str:
    """Synthesize text to speech using edge-tts and save as MP3.

    Args:
        text: The text to convert to speech.
        output_path: File path to save the generated .mp3 audio.

    Returns:
        The file path of the saved audio file.
    """
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await communicate.save(output_path)
    return output_path
