# RAG Study Assistant Dockerfile
# GPU-enabled container for local Ollama + FastAPI/Gradio

FROM python:3.12-slim

LABEL maintainer="Raktim"
LABEL description="RAG Study Assistant with local LLM"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV OLLAMA_MODEL=llama3.2:3b
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV CHROMA_DB_PATH=/app/chroma_db

EXPOSE 7860 8000

CMD ["python", "app.py"]