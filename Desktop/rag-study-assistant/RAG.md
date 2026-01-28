\# RAG Study Assistant



An AI-powered study assistant that lets you upload PDFs and ask questions about them using Retrieval Augmented Generation (RAG).



\## Features

\- Upload multiple PDF files

\- Ask questions about your study materials

\- Get accurate answers based on your documents

\- Runs locally on your GPU (NVIDIA 3050 6GB)

\- Completely private - no data leaves your computer



\## Tech Stack

\- Python

\- Ollama (Llama 3.2 3B)

\- Gradio (Web Interface)

\- ChromaDB (Vector Database)

\- PyPDF (PDF Processing)

\- Sentence Transformers (Embeddings)



\## Installation



\### Prerequisites

1\. Install Ollama from https://ollama.ai

2\. Python 3.8 or higher



\### Setup



1\. Clone this repository

2\. Create virtual environment:

```bash

python -m venv venv

venv\\Scripts\\activate

```



3\. Install dependencies:

```bash

pip install -r requirements.txt

```



4\. Download the AI model:

```bash

ollama pull llama3.2:3b

```



\## Usage



1\. Start the application:

```bash

python app.py

```



2\. Open browser at http://localhost:7860



3\. Upload your PDF files (textbooks, notes, research papers)



4\. Ask questions about the content!



\## How It Works



1\. \*\*PDF Processing\*\*: Extracts text from uploaded PDFs

2\. \*\*Chunking\*\*: Splits text into manageable chunks

3\. \*\*Embedding\*\*: Converts chunks into vector embeddings

4\. \*\*Vector Storage\*\*: Stores in ChromaDB for fast retrieval

5\. \*\*Semantic Search\*\*: Finds relevant chunks for user questions

6\. \*\*AI Generation\*\*: Uses Llama 3.2 to generate answers based on context



\## Project Structure

```

rag-study-assistant/

├── app.py              # Main application

├── pdf\_reader.py       # PDF processing \& RAG logic

├── test\_pdf.py         # Testing script

├── requirements.txt    # Dependencies

└── README.md          # Documentation

```



\## Future Enhancements

\- \[ ] Conversation history

\- \[ ] Citation/source highlighting

\- \[ ] Support for Word documents

\- \[ ] Multi-language support

\- \[ ] Export chat history



\## Author

Computer Science Engineering Student

Built as part of learning AI/ML engineering



\## License

MIT License

