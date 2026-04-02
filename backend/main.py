import os
import uuid
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

from backend.stt import transcribe_audio
from backend.llm import generate_response
from backend.tts import synthesize_speech
from backend.qdrant_service import initialize_collection, search

# Directory to store generated audio files
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Qdrant collection on startup."""
    print("[Startup] Initializing Qdrant collection...")
    initialize_collection()
    print("[Startup] Qdrant collection ready.")
    yield


app = FastAPI(
    title="Voice AI Support Assistant",
    description="E-commerce voice support powered by Groq, Qdrant, and edge-tts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
async def handle_query(audio: UploadFile = File(...)):
    """Full voice support pipeline: STT → RAG → LLM → TTS.

    Accepts a multipart audio file upload and returns:
    - transcript: the transcribed user speech
    - response: the LLM-generated answer
    - audio_url: URL to the synthesized speech audio
    """
    # Step 1: Save uploaded audio to a temp file
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 2: Speech-to-Text
        transcript = transcribe_audio(tmp_path)

        # Step 3: Qdrant semantic search for relevant context
        context_chunks = search(transcript, top_k=3)

        # Step 4: LLM generates response with RAG context
        response_text = generate_response(transcript, context_chunks)

        # Step 5: Text-to-Speech
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        await synthesize_speech(response_text, audio_path)

        return JSONResponse(
            content={
                "transcript": transcript,
                "response": response_text,
                "audio_url": f"/audio/{audio_filename}",
            }
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve a generated audio file."""
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
