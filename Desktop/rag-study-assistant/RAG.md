# 📚 RAG Study Assistant (Advanced)

An AI-powered study assistant that uses Retrieval Augmented Generation (RAG) to answer questions from your PDF documents with conversation memory and context awareness.

![Demo Screenshot](screenshot1.png)

## ✨ Features

- 📤 **Multi-PDF Upload** - Process multiple documents simultaneously
- 💬 **Conversation Memory** - Remembers context from previous questions
- 🔍 **Semantic Search** - Finds relevant information using vector embeddings
- 📊 **Document Analytics** - Shows loaded files, character counts, page numbers
- 💾 **Export Conversations** - Save your Q&A sessions for later reference
- 🎯 **Source Citations** - Indicates answers are from your documents
- ⚡ **GPU Accelerated** - Runs on local NVIDIA GPU for speed and privacy
- 🧹 **Clear Session** - Start fresh anytime

## 🎥 Demo

[Video Demo](link-to-your-video-when-ready)

## 🛠️ Tech Stack

- **AI Model:** Llama 3.2 (3B parameters) via Ollama
- **Vector Database:** ChromaDB with sentence transformers
- **Frontend:** Gradio 6.0
- **PDF Processing:** PyPDF
- **Language:** Python 3.12

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU (tested on RTX 3050 6GB)
- Ollama installed locally

## 🚀 Installation

### 1. Install Ollama
Download from [ollama.ai](https://ollama.ai) and install.

### 2. Pull the AI model
```bash
ollama pull llama3.2:3b
```

### 3. Clone and setup
```bash
git clone https://github.com/rktm0604/RAG.git
cd RAG
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Start the application
```bash
python app.py
```

### Open in browser
Navigate to `http://localhost:7860`

### Upload and Ask
1. Click "Select PDF files" and choose your documents
2. Click "Process PDFs" and wait for confirmation
3. Ask questions in the chat interface
4. Get AI-powered answers based on your documents!

## 📸 Screenshots

### Upload Interface
![Upload](screenshot1.png)

### Chat Interface
![Chat](screenshot2.png)

### Document Info
![Info](screenshot3.png)

## 🔧 How It Works

1. **Document Processing**
   - PDFs are read and text is extracted
   - Text is split into 1000-character chunks

2. **Vector Embeddings**
   - Chunks are converted to vector embeddings
   - Stored in ChromaDB for fast retrieval

3. **Query Processing**
   - User question is embedded
   - Similar chunks are retrieved (top 3)
   - Context is provided to the LLM

4. **Answer Generation**
   - Llama 3.2 generates answer using context
   - Conversation history is maintained
   - Source citation is added

## 🎯 Use Cases

- 📖 Study from textbooks and lecture notes
- 📄 Research paper analysis
- 📋 Document summarization
- 🔍 Quick information lookup
- 📝 Exam preparation

## 🔒 Privacy

- All processing happens locally on your machine
- No data is sent to external servers
- Your documents stay private
- Completely offline after model download

## ⚙️ Configuration

Model can be changed in `app.py`:
```python
model='llama3.2:3b'  # Change to other Ollama models
```

Chunk size adjustable in `pdf_reader.py`:
```python
chunk_size = 1000  # Modify as needed
```

## 🐛 Troubleshooting

**Ollama connection error:**
- Ensure Ollama is running: `ollama serve`
- Check model is downloaded: `ollama list`

**PDF not loading:**
- Check file isn't password protected
- Ensure PDF contains extractable text (not scanned images)

**Out of memory:**
- Try smaller PDFs
- Reduce chunk size
- Use lighter model like `phi3:mini`

## 🚧 Future Enhancements

- [ ] Support for Word documents (.docx)
- [ ] Multi-language support
- [ ] Page number citations
- [ ] Chat history persistence
- [ ] Web deployment option
- [ ] Dark mode toggle

## 👨‍💻 Author

**Your Name**  
Computer Science Engineering Student  
GitHub: [@rktm0604](https://github.com/rktm0604)

## 📝 License

MIT License

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai)
- UI powered by [Gradio](https://gradio.app)
- Vector storage by [ChromaDB](https://www.trychroma.com)

---

⭐ Star this repo if you find it useful!
