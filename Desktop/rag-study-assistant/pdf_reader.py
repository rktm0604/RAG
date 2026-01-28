from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

def load_pdf(pdf_path):
    """Extract text from PDF"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_knowledge_base(text):
    """Split text into chunks and create searchable database"""
    # Split into chunks of 1000 characters
    chunks = []
    chunk_size = 1000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    # Create vector database
    client = chromadb.Client()
    
    # Delete collection if it exists (for fresh start)
    try:
        client.delete_collection(name="study_docs")
    except:
        pass
    
    collection = client.create_collection(
        name="study_docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
    )
    
    # Add chunks to database
    if chunks:
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
    
    return collection

def search_knowledge(collection, query):
    """Search for relevant information"""
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    if results['documents'] and results['documents'][0]:
        return "\n\n".join(results['documents'][0])
    return "No relevant information found."