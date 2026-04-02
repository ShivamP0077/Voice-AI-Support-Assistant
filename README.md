# 🛒 Voice AI Support Assistant

An intelligent voice-powered customer support assistant for e-commerce platforms. Speak your question, and the AI will respond with accurate answers using retrieval-augmented generation (RAG) — both as text and synthesized speech.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20%2B%20Whisper-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-blue)
![edge--tts](https://img.shields.io/badge/edge--tts-TTS-green)

## ✨ Features

- 🎙️ **Voice Input** — Record your question using the browser microphone
- 🧠 **RAG Pipeline** — Retrieves relevant order/policy context before answering
- ⚡ **Groq Whisper STT** — Fast, accurate speech-to-text transcription
- 🤖 **LLM Response** — Grounded answers via Llama 3 (8B) on Groq
- 🔊 **Natural TTS** — AI response read aloud using edge-tts (Jenny Neural voice)
- 🔍 **Semantic Search** — Qdrant vector store with sentence-transformer embeddings

## 🏗️ Architecture

```
Browser (mic) → Audio Upload → FastAPI Backend
                                    │
                                    ├─ STT (Groq Whisper) → transcript
                                    ├─ Vector Search (Qdrant) → context chunks
                                    ├─ LLM (Groq Llama 3) → response text
                                    └─ TTS (edge-tts) → audio file
                                    │
                                    ▼
                              JSON Response
                    { transcript, response, audio_url }
```

## 📁 Project Structure

```
voice-ai-support-assistant/
├── backend/
│   ├── main.py               # FastAPI app with full pipeline
│   ├── stt.py                # Speech-to-text (Groq Whisper)
│   ├── llm.py                # LLM response generation (Groq)
│   ├── tts.py                # Text-to-speech (edge-tts)
│   ├── qdrant_service.py     # Qdrant vector search
│   ├── data_loader.py        # Load + embed orders & policies
│   └── config.py             # API keys, constants
├── data/
│   ├── orders.json           # Sample e-commerce orders
│   └── policies.json         # Store policies (return, shipping, etc.)
├── frontend/
│   └── index.html            # Browser UI with mic recording
├── .env                      # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)
- Internet connection (required for Groq API and edge-tts)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/voice-ai-support-assistant.git
   cd voice-ai-support-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file** in the project root
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Start the backend server**
   ```bash
   uvicorn backend.main:app --reload
   ```

6. **Open the frontend**
   
   Open `frontend/index.html` in your browser (or serve it via a local server).

### Usage

1. Click the 🎙️ microphone button to start recording
2. Ask a question about orders, returns, shipping, or policies
3. Click again to stop recording
4. Wait for the AI to process your query
5. View the transcript, read the AI response, and listen to the audio

## 📋 Assumptions

- Audio input is in `webm` format from the browser's MediaRecorder API
- Qdrant runs **in-memory** (no persistence needed for this demo)
- User identity is not authenticated; orders are matched by semantic search on query content
- edge-tts requires an **internet connection** to synthesize speech
- The Groq API key must have access to both `whisper-large-v3` and `llama3-8b-8192` models

## 🧠 Design Decisions & Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **Qdrant in-memory** | Avoids Docker/infrastructure dependency for easy local demo |
| **Groq Whisper** over local Whisper | Significantly faster inference via cloud API |
| **edge-tts** over pyttsx3 | Much more natural-sounding voices (Jenny Neural); pyttsx3 is an offline fallback option |
| **RAG over orders + policies** | Keeps LLM responses grounded in real data, reduces hallucination |
| **Sentence-Transformers (all-MiniLM-L6-v2)** | Lightweight, fast, good quality embeddings (384 dimensions) |
| **No frontend framework** | Simple single-page HTML/CSS/JS — minimal complexity, easy to deploy |
| **CORS allow all** | Development convenience; should be restricted in production |

## 🔮 Future Improvements

- 🔐 **User authentication** to filter orders by logged-in user
- 💾 **Persist Qdrant to disk** for production deployments
- 🌊 **Stream LLM responses** for lower perceived latency
- 🎤 **Fallback STT** using browser Web Speech API when Groq is unavailable
- 🔄 **CI/CD pipeline** for automated testing and deployment
- 📊 **Conversation history** to support multi-turn interactions
- 🌐 **Multi-language support** using different edge-tts voices
- 📱 **PWA support** for mobile installation

## 🛠️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Upload audio file → returns transcript, response, and audio URL |
| `GET` | `/audio/{filename}` | Serves generated TTS audio files |

### POST /query

**Request:** Multipart form data with `audio` file field

**Response:**
```json
{
  "transcript": "What is your return policy?",
  "response": "Items can be returned within 7 days of delivery...",
  "audio_url": "/audio/abc123.mp3"
}
```

## 📄 License

This project is for demonstration purposes.