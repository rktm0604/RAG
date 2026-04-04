"""
FastAPI wrapper for RAG Study Assistant
Wraps the existing app.py logic into a REST API

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /upload     — Upload and process PDF files
    POST /ask        — Ask a question against loaded documents
    GET  /status     — Check if knowledge base is loaded
    POST /clear      — Reset the knowledge base
    GET  /history    — Get conversation history
    GET  /health     — Health check (Ollama + model status)
"""

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pdf_reader import (
    create_knowledge_base,
    load_pdf,
    reset_knowledge_base,
    search_knowledge,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Study Assistant API",
    description="REST API wrapper for the RAG Study Assistant — ask questions against your PDF documents.",
    version="1.0.0",
)

# Allow all origins (useful for Azure deployment / frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session state (single user — extend to Redis for multi-user)
# ---------------------------------------------------------------------------

session = {
    "knowledge_base": None,
    "files": [],
    "conversation": [],
}

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    stream: bool = False  # Set True for streaming response

class AskResponse(BaseModel):
    answer: str
    source_pages: list[int]
    model: str
    timestamp: str

class UploadResponse(BaseModel):
    message: str
    files_loaded: list[dict]
    total_pages: int

class StatusResponse(BaseModel):
    knowledge_base_loaded: bool
    files_loaded: list[dict]
    conversation_count: int
    model: str

class HealthResponse(BaseModel):
    ollama_running: bool
    model_available: bool
    model: str

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check if Ollama is running and the model is available."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, "models") else []
        model_ok = any(OLLAMA_MODEL in m for m in available)
        return HealthResponse(
            ollama_running=True,
            model_available=model_ok,
            model=OLLAMA_MODEL,
        )
    except Exception as e:
        logger.error("Ollama health check failed: %s", e)
        return HealthResponse(
            ollama_running=False,
            model_available=False,
            model=OLLAMA_MODEL,
        )

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse, tags=["Session"])
def get_status():
    """Check current session status — is a knowledge base loaded?"""
    return StatusResponse(
        knowledge_base_loaded=session["knowledge_base"] is not None,
        files_loaded=session["files"],
        conversation_count=len(session["conversation"]),
        model=OLLAMA_MODEL,
    )

# ---------------------------------------------------------------------------
# Upload PDFs
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDF files and build the knowledge base.
    Accepts multipart/form-data with multiple PDF files.

    Example curl:
        curl -X POST http://localhost:8000/upload \\
          -F "files=@study_notes.pdf" \\
          -F "files=@chapter2.pdf"
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    all_pages = []
    uploaded_info = []
    tmp_dir = tempfile.mkdtemp()

    try:
        for upload in files:
            if not upload.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{upload.filename}' is not a PDF.",
                )

            # Save to temp file so pdf_reader can process it
            tmp_path = Path(tmp_dir) / upload.filename
            with open(tmp_path, "wb") as f:
                content = await upload.read()
                f.write(content)

            logger.info("Processing: %s", upload.filename)
            pages = load_pdf(str(tmp_path))
            all_pages.extend(pages)

            uploaded_info.append({
                "name": upload.filename,
                "pages": len(pages),
                "chars": sum(len(t) for _, t in pages),
            })

        # Build ChromaDB knowledge base
        collection = create_knowledge_base(all_pages)
        total_pages = sum(info["pages"] for info in uploaded_info)

        # Update session
        session["knowledge_base"] = collection
        session["files"] = uploaded_info
        session["conversation"] = []

        logger.info("Knowledge base built from %d file(s), %d pages total.", len(files), total_pages)

        return UploadResponse(
            message=f"Successfully loaded {len(files)} PDF(s) with {total_pages} pages total.",
            files_loaded=uploaded_info,
            total_pages=total_pages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing PDFs: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Ask a question (non-streaming)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(body: AskRequest):
    """
    Ask a question against the loaded knowledge base.
    Returns the answer with source page citations.

    Example curl:
        curl -X POST http://localhost:8000/ask \\
          -H "Content-Type: application/json" \\
          -d '{"question": "What is the main topic of the document?"}'
    """
    if session["knowledge_base"] is None:
        raise HTTPException(
            status_code=400,
            detail="No knowledge base loaded. Upload PDFs first via POST /upload.",
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Retrieve context + source pages from ChromaDB
        context, source_pages = search_knowledge(session["knowledge_base"], body.question)

        # Build conversation history context (last 3 exchanges)
        history_text = ""
        recent = session["conversation"][-3:]
        if recent:
            history_text = "\nRecent conversation:\n"
            for entry in recent:
                history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n"

        prompt = f"""You are a helpful study assistant. Answer the question based on the provided context from uploaded documents.

Context from documents:
{context}
{history_text}

Current question: {body.question}

Instructions:
1. Answer based primarily on the provided context
2. Be specific and cite information from the context
3. If the context doesn't contain the answer, say so clearly
4. Keep answers concise but complete
5. Use the conversation history for context if relevant

Answer:"""

        # Call Ollama (non-streaming)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        answer = response["message"]["content"]

        # Save to conversation history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        session["conversation"].append({
            "question": body.question,
            "answer": answer,
            "pages": source_pages,
            "timestamp": timestamp,
        })

        logger.info("Q: %s | Pages cited: %s", body.question[:60], source_pages)

        return AskResponse(
            answer=answer,
            source_pages=source_pages,
            model=OLLAMA_MODEL,
            timestamp=timestamp,
        )

    except Exception as e:
        logger.error("Error generating response: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# ---------------------------------------------------------------------------
# Ask a question (streaming)
# ---------------------------------------------------------------------------

@app.post("/ask/stream", tags=["Q&A"])
def ask_question_stream(body: AskRequest):
    """
    Ask a question with streaming response (Server-Sent Events).
    Streams tokens as they are generated — same as the Gradio UI experience.

    Example curl:
        curl -X POST http://localhost:8000/ask/stream \\
          -H "Content-Type: application/json" \\
          -d '{"question": "Summarise the document"}' \\
          --no-buffer
    """
    if session["knowledge_base"] is None:
        raise HTTPException(
            status_code=400,
            detail="No knowledge base loaded. Upload PDFs first via POST /upload.",
        )

    context, source_pages = search_knowledge(session["knowledge_base"], body.question)

    history_text = ""
    recent = session["conversation"][-3:]
    if recent:
        history_text = "\nRecent conversation:\n"
        for entry in recent:
            history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n"

    prompt = f"""You are a helpful study assistant. Answer the question based on the provided context.

Context:
{context}
{history_text}

Question: {body.question}

Answer:"""

    def token_generator():
        full_response = ""
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                full_response += token
                yield token

        # Save after streaming completes
        session["conversation"].append({
            "question": body.question,
            "answer": full_response,
            "pages": source_pages,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return StreamingResponse(token_generator(), media_type="text/plain")

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

@app.get("/history", tags=["Session"])
def get_history():
    """Get full conversation history for the current session."""
    return {
        "count": len(session["conversation"]),
        "conversation": session["conversation"],
    }

# ---------------------------------------------------------------------------
# Clear session
# ---------------------------------------------------------------------------

@app.post("/clear", tags=["Session"])
def clear_session():
    """Reset the knowledge base and conversation history."""
    reset_knowledge_base()
    session["knowledge_base"] = None
    session["files"] = []
    session["conversation"] = []
    logger.info("Session cleared via API.")
    return {"message": "Session cleared. Upload new PDFs to start fresh."}

# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
