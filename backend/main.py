import os
import uuid
import logging
import tempfile
import json

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from backend.stt import transcribe_audio
from backend.llm import generate_response, generate_response_stream
from backend.tts import synthesize_speech
from backend.qdrant_service import initialize_collection, search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store generated audio files
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Qdrant collection on startup."""
    logger.info("Initializing Qdrant collection...")
    initialize_collection()
    logger.info("Qdrant collection ready.")
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))


@app.post("/query")
async def handle_query(audio: UploadFile = File(...)):
    """Full voice support pipeline: STT → RAG → LLM → TTS.

    Accepts a multipart audio file upload and returns:
    - transcript: the transcribed user speech
    - response: the LLM-generated answer
    - audio_url: URL to the synthesized speech audio
    """
    tmp_path = None

    try:
        # Step 1: Save uploaded audio to a temp file
        suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Received audio file: {audio.filename} ({len(content)} bytes)")

    except Exception as e:
        logger.error(f"Failed to save uploaded audio: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to process uploaded audio file."},
        )

    try:
        # Step 2: Speech-to-Text
        logger.info("Running speech-to-text...")
        transcript = transcribe_audio(tmp_path)
        logger.info(f"Transcript: {transcript}")
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Speech-to-text transcription failed. Please try again."},
        )

    try:
        # Step 3: Qdrant semantic search for relevant context (hardcoded to U1 prototype)
        logger.info("Searching for relevant context...")
        context_chunks = search(transcript, top_k=3, user_id="U1")
        logger.info(f"Found {len(context_chunks)} context chunks.")
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Context retrieval failed. Please try again."},
        )

    try:
        # Step 4: LLM generates response with RAG context
        logger.info("Generating LLM response...")
        response_text = generate_response(transcript, context_chunks)
        logger.info(f"LLM response length: {len(response_text)} chars")
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Response generation failed. Please try again."},
        )

    try:
        # Step 5: Text-to-Speech
        logger.info("Synthesizing speech...")
        audio_filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        await synthesize_speech(response_text, audio_path)
        logger.info(f"Audio saved: {audio_filename}")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Speech synthesis failed. Please try again."},
        )

    # Clean up temp file
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return JSONResponse(
        content={
            "transcript": transcript,
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}",
        }
    )


@app.post("/api/v2/query")
async def handle_query_v2_stream(audio: UploadFile = File(...)):
    """V2 voice support pipeline with Server-Sent Events (SSE) streaming."""
    tmp_path = None

    try:
        suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"V2 Request: Received audio file: {audio.filename} ({len(content)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save uploaded audio: {e}")
        return JSONResponse(status_code=500, content={"detail": "Failed to process audio."})

    async def event_generator():
        try:
            # 1. STT
            transcript = transcribe_audio(tmp_path)
            yield f"event: transcript\ndata: {json.dumps({'text': transcript})}\n\n"

            # 2. Qdrant Search (mock auth for prototype)
            context_chunks = search(transcript, top_k=3, user_id="U1")

            # 3. LLM Streaming
            full_response = ""
            for chunk in generate_response_stream(transcript, context_chunks):
                full_response += chunk
                yield f"event: text\ndata: {json.dumps({'chunk': chunk})}\n\n"

            # 4. Generate TTS (after text finishes)
            audio_filename = f"{uuid.uuid4().hex}.mp3"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            await synthesize_speech(full_response, audio_path)
            yield f"event: audio\ndata: {json.dumps({'url': f'/audio/{audio_filename}'})}\n\n"

            # End of stream
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve a generated audio file."""
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
