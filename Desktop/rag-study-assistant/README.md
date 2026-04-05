<div align="center">

<img src="https://img.shields.io/badge/AI--Powered-Study_Assistant-6C63FF?style=for-the-badge&logo=robot&logoColor=white" alt="RAG Study Assistant" />

# 📚 RAG Study Assistant

### *Your intelligent study companion that actually knows your documents.*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-000000?style=for-the-badge&logo=meta&logoColor=white)](https://ollama.ai)
[![Gradio](https://img.shields.io/badge/Gradio-6.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-4A154B?style=for-the-badge)](https://www.trychroma.com)
[![Pytest](https://img.shields.io/badge/tested_with-pytest-0A9EDC?style=for-the-badge&logo=pytest)](https://docs.pytest.org/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Upload PDFs → Ask Questions → Get AI-Powered Answers — 100% Local & Private**

[✨ Features](#-features) • [📸 Demo](#-demo) • [🚀 Quick Start](#-quick-start) • [🔧 Setup OCR](#-optional-setup-ocr-for-scanned-pdfs) • [🧠 Under the Hood](#-under-the-hood)

</div>

---

## 🎯 What Is This?

A **Retrieval Augmented Generation (RAG)** system that turns your raw PDF files into an interactive, intelligent knowledge base. Upload textbooks, research papers, or lecture notes, and simply ask questions. The AI finds relevant passages from your exact documents and generates accurate, context-aware answers with **page citations**.

> **Built for students, researchers, and continuous learners.**

### 💎 Why This Project?
- 🔐 **100% Private** — All processing happens locally on your GPU. Nothing leaves your machine.
- 🧠 **Smart Retrieval** — Uses advanced semantic search with vector embeddings, not just Ctrl+F keyword matching.
- 📄 **Precise Citations** — Answers cite the exact page numbers they were drawn from in your PDFs.
- 🔤 **OCR Support** — Can read scanned and image-based PDFs automatically using Tesseract.
- ⚡ **Real-time Streaming** — Responses stream in as they are generated for a snappy, ChatGPT-like experience.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **Multi-PDF Upload** | Process multiple textbooks or papers simultaneously into one cohesive knowledge base. |
| 📄 **Page-Number Citations** | The AI tells you exactly which page(s) it used to generate the answer. |
| 🔤 **OCR Fallback** | Automatically attempts OCR via Tesseract/Poppler if a PDF has no selectable text. |
| ⚡ **Streaming Chat** | Fast, real-time typing effect during generation. |
| 💬 **Conversation Memory** | Remembers context from previous questions for natural follow-ups. |
| 💾 **Export Conversations** | Save your Q&A sessions (with citations!) as Markdown for your notes. |
| 💿 **Persistent Storage** | ChromaDB persists your vectorized docs so you don't rebuild them on restart. |
| ⚙️ **Configurable Models** | Easily swap out LLMs via the `OLLAMA_MODEL` environment variable. |
| 🧠 **State-of-the-art Embeddings** | Uses BAAI/bge-small-en-v1.5 for superior semantic search quality |
| 🔀 **Hybrid Search** | Combines BM25 keyword search with semantic vector search |
| 🎯 **Reranking** | CoHERE reranker improves result ranking accuracy |
| 📊 **RAG Evaluation** | Built-in metrics to measure answer quality (precision, faithfulness, relevance) |

---

## 📸 Demo

<div align="center">

### Dashboard - Modern Card-Based UI
![Dashboard](Screenshot%202026-02-01%20183405.png)

### RAG Chat Interface
![Chat Interface](Screenshot%2022026-02-01%20184205.png)

</div>

---

## 🎨 Dashboard UI

The project features a **modern, responsive dashboard** built with Gradio 6.0 Blocks API:

- **Navbar** — Logo, navigation, search, and Get Started button
- **Hero Section** — Animated gradient background
- **Feature Cards** — Interactive cards showcasing:
  - 🔍 Semantic Search (BGE embeddings)
  - ⚡ Hybrid Search (BM25 + Vector)
  - 🎯 AI Reranking (CoHERE)
- **Split Layout** — Upload panel + Chat interface
- **Modern Styling** — Soft shadows, rounded corners, hover effects, responsive design

### Running the Dashboard

```bash
python app.py
```

Open `http://localhost:7860`

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU** (Tested on RTX 3050 6GB; significantly improves inference speed)
- **Ollama** installed from [ollama.ai](https://ollama.ai)
- **Docker & Docker Compose** (optional, for containerized deployment)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rktm0604/RAG.git
cd RAG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install python dependencies
pip install -r requirements.txt

# Download the default AI model (Llama 3.2 3B)
ollama pull llama3.2:3b
```

### 2. Run the Assistant

```bash
# Ensure Ollama is running in the background, then start the web app:
python app.py
```
*Open `http://localhost:7860` in your browser!*

### 3. (Optional) Docker Deployment

```bash
# With GPU support
docker-compose up --build

# Or just the app (without Ollama container)
docker build -t rag-assistant .
docker run -p 7860:7860 -p 8000:8000 rag-assistant
```

---

## 🔤 (Optional) Setup OCR for Scanned PDFs

If you plan to upload scanned book pages or image-based PDFs, the app will gracefully fall back to OCR. You just need two system dependencies:

1. **Install Tesseract OCR**
   - **Windows:** Download the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Run it, and ensure `C:\Program Files\Tesseract-OCR` is added to your system `PATH`.
   - **Linux:** `sudo apt install tesseract-ocr`
   - **macOS:** `brew install tesseract`

2. **Install Poppler** (required by pdf2image)
   - **Windows:** Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add the `bin/` folder to your system `PATH`.
   - **Linux:** `sudo apt install poppler-utils`
   - **macOS:** `brew install poppler`

---

## 🧪 Testing & Evaluation

### Unit Tests
```bash
python -m pytest tests/ -v
```

### RAG Evaluation Metrics
The project includes RAGAS-style evaluation to measure answer quality:

```bash
# Single evaluation
python -m rag_eval --question "What is photosynthesis?" --answer "..." --context "..."

# Batch evaluation
python -c "
from rag_eval import evaluate_batch
examples = [
    {'question': '...', 'answer': '...', 'context': '...'},
]
print(evaluate_batch(examples))
"
```

Metrics tracked:
- **Context Precision** — How relevant is the retrieved context?
- **Faithfulness** — Does the answer match the context?
- **Answer Relevance** — Does the answer address the question?

---

## ⚙️ Advanced Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer model |
| `EMBEDDING_DEVICE` | `cuda` | Device for embeddings (`cpu` or `cuda`) |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector database location |
| `COHERE_API_KEY` | — | Required for reranking (get free key at cohere.com) |

### Setting Up CoHERE API Key (for reranking)

Create a `.env` file in the project root:

```bash
COHERE_API_KEY=your-api-key-here
```

Get a free API key at: https://dashboard.cohere.com/api-keys

### Advanced Search Options

The `search_knowledge` function supports:
- **Hybrid Search** — Combine BM25 + semantic search: `search_knowledge(..., use_hybrid=True)`
- **Reranking** — Use CoHERE for improved ranking: `search_knowledge(..., use_reranker=True)`

## 🧠 Under the Hood

### The Architecture Work-flow

```mermaid
graph TD
    A[📄 User Uploads PDEs] --> B{Is Text Selectable?}
    B -->|Yes| C[PyPDF Extraction]
    B -->|No| D[Tesseract OCR Fallback]
    C --> E[✂️ Smart Chunking w/ Overlap & Page Tracking]
    D --> E
    E --> F[🔢 BGE Embeddings (bge-small-en-v1.5)]
    F --> G[(💾 Persistent ChromaDB Vector Store)]
    
    H[❓ User Asks Question] --> I[🔢 Embed Query]
    I --> J{🔍 Search Mode?}
    J -->|Semantic| K[ChromaDB Semantic Search]
    J -->|Hybrid| L[BM25 + Semantic Combined]
    J -->|Rerank| M[CoHERE Reranker]
    G --> K
    G --> L
    K --> M
    L --> M
    M -->|Returns Top Chunks + Page #s| N[🤖 Local Llama 3.2 Model]
    N -->|Streams Output| O[💬 Answer with Page Citations]
    
    P[📊 RAG Evaluation] -.->|Measures| O
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Llama 3.2 (3B) via Ollama | Answer generation & reasoning |
| **Vector DB** | ChromaDB (Persistent) | Vector embedding storage & semantic search |
| **Embeddings** | BAAI/bge-small-en-v1.5 | State-of-the-art semantic embeddings |
| **Reranking** | CoHERE rerank-multilingual-v3 | Improved result ranking |
| **Hybrid Search** | BM25 + Semantic | Combined keyword + vector search |
| **Evaluation** | RAGAS-style metrics | Quality measurement |
| **PDF Toolkit** | PyPDF + pytesseract | Text extraction and image OCR fallback |
| **Frontend UI** | Gradio 6.0 | Web-based chat & upload interface |
| **API** | FastAPI | REST endpoints for integration |
| **Deployment** | Docker + Compose | Containerized production-ready setup |
| **Testing** | Pytest | Ensuring chunking, metadata, and pipeline integrity |

---

## 🚧 Roadmap

- [x] Page number citations in answers
- [x] Streaming response generation
- [x] OCR support for scanned documents
- [x] RAG evaluation metrics
- [x] Advanced embeddings (bge-small-en-v1.5)
- [x] Hybrid search (BM25 + semantic)
- [x] Reranking with CoHERE
- [x] Docker deployment
- [ ] Support for Word documents (`.docx`)
- [ ] Multi-language support
- [ ] Web deployment (Hugging Face Spaces)

---

## 👨‍💻 Author

**Raktim** — Computer Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-@rktm0604-181717?style=for-the-badge&logo=github)](https://github.com/rktm0604)

---

<div align="center">

**If this project helps you study smarter or ace your exams, give it a ⭐!**  
Released under the [MIT License](LICENSE).

</div>
