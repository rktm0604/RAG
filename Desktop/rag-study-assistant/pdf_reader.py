"""RAG pipeline — PDF loading, chunking with page metadata, and vector search."""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "study_docs"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

import torch
os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"


def _get_embedding_function():
    """Returns BGE embedding function via ChromaDB's built-in wrapper."""
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL, device="cuda")


def create_knowledge_base(pages, chunk_size=1000, chunk_overlap=200):


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdf(pdf_path):
    """Extract text from a PDF file, preserving page numbers.

    Tries PyPDF text extraction first. If the PDF is scanned / image-based
    and no text is found, falls back to OCR via pytesseract + pdf2image.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples (1-indexed).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted even with OCR.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: '{pdf_path}'")

    # --- Attempt 1: native text extraction via PyPDF ---
    reader = PdfReader(pdf_path)
    pages = []

    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append((idx, page_text))

    if pages:
        total_chars = sum(len(t) for _, t in pages)
        logger.info(
            "Loaded PDF '%s' (text mode): %d pages, %d chars",
            path.name, len(pages), total_chars,
        )
        return pages

    # --- Attempt 2: OCR fallback for scanned / image-based PDFs ---
    if not OCR_AVAILABLE:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF appears to be scanned or image-based.\n\n"
            "To enable OCR, install the optional dependencies:\n"
            "  pip install pytesseract pdf2image\n"
            "and install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
        )

    logger.info("No text found via PyPDF — falling back to OCR for '%s'", path.name)

    try:
        images = convert_from_path(str(path))
    except Exception as e:
        raise ValueError(
            f"OCR failed for '{path.name}': {e}\n\n"
            "Make sure Poppler is installed and on your PATH:\n"
            "  • Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  • Linux:   sudo apt install poppler-utils\n"
            "  • macOS:   brew install poppler"
        ) from e

    pages = []
    for idx, image in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(image)
        if page_text and page_text.strip():
            pages.append((idx, page_text))

    if not pages:
        raise ValueError(
            f"No text could be extracted from '{path.name}' even with OCR. "
            "The PDF may be blank or contain unrecognizable content."
        )

    total_chars = sum(len(t) for _, t in pages)
    logger.info(
        "Loaded PDF '%s' (OCR mode): %d pages, %d chars",
        path.name, len(pages), total_chars,
    )
    return pages


def pages_to_text(pages):
    """Flatten a list of (page_number, page_text) tuples into a single string.

    Convenience helper for callers that only need the raw text.
    """
    return "\n".join(text for _, text in pages)


# ---------------------------------------------------------------------------
# Chunking with page metadata
# ---------------------------------------------------------------------------

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks, breaking on sentence boundaries.

    Args:
        text: The input text to split.
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we haven't reached the end, try to break at a sentence boundary
        if end < len(text):
            search_start = max(start, end - 200)
            search_region = text[search_start:end]
            for sep in ['. ', '.\n', '? ', '!\n', '?\n', '! ', '\n\n']:
                last_break = search_region.rfind(sep)
                if last_break != -1:
                    end = search_start + last_break + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap), at least 1 char to avoid loops
        step = max(1, chunk_size - chunk_overlap)
        start = start + step if end >= start + chunk_size else end

    logger.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks


def chunk_text_with_pages(pages, chunk_size=1000, chunk_overlap=200):
    """Split page-annotated text into overlapping chunks with page metadata.

    Args:
        pages: List of (page_number, page_text) tuples from load_pdf.
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of dicts with keys: 'text', 'pages' (sorted list of page numbers).
    """
    if not pages:
        return []

    # Build a character-offset → page-number mapping
    full_text = ""
    char_to_page = []  # parallel array: char_to_page[i] = page number for char i

    for page_num, page_text in pages:
        page_with_separator = page_text + "\n"
        for _ in page_with_separator:
            char_to_page.append(page_num)
        full_text += page_with_separator

    # Get text chunks
    raw_chunks = chunk_text(full_text, chunk_size, chunk_overlap)

    # Map each chunk back to its source pages
    result = []
    search_start = 0

    for chunk in raw_chunks:
        idx = full_text.find(chunk, search_start)
        if idx == -1:
            # Fallback: search from beginning (should rarely happen)
            idx = full_text.find(chunk)

        if idx != -1:
            chunk_pages = set()
            end_idx = min(idx + len(chunk), len(char_to_page))
            for ci in range(idx, end_idx):
                chunk_pages.add(char_to_page[ci])
            result.append({
                "text": chunk,
                "pages": sorted(chunk_pages),
            })
            search_start = idx + 1
        else:
            # If we can't locate the chunk (edge case), include without pages
            result.append({"text": chunk, "pages": []})

    logger.info(
        "Created %d page-annotated chunks (size=%d, overlap=%d)",
        len(result), chunk_size, chunk_overlap,
    )
    return result


# ---------------------------------------------------------------------------
# ChromaDB operations
# ---------------------------------------------------------------------------

def get_chroma_client():
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def create_knowledge_base(pages, chunk_size=1000, chunk_overlap=200):
    """Build a vector knowledge base from page-annotated text.

    Args:
        pages: Either a list of (page_number, page_text) tuples **or** a plain
               string (for backward compatibility).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        ChromaDB collection for querying.
    """
    request_id = str(uuid.uuid4())[:8]
    
    if isinstance(pages, str):
        if not pages.strip():
            raise ValueError("No text chunks were created. The document may be empty.")
        raw_chunks = chunk_text(pages, chunk_size, chunk_overlap)
        if not raw_chunks:
            raise ValueError("No text chunks were created. The document may be empty.")
        chunked = [{"text": c, "pages": []} for c in raw_chunks]
    else:
        chunked = chunk_text_with_pages(pages, chunk_size, chunk_overlap)
        if not chunked:
            raise ValueError("No text chunks were created. The document may be empty.")

    client = get_chroma_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
    )

    collection.add(
        documents=[c["text"] for c in chunked],
        metadatas=[{"pages": ",".join(str(p) for p in c["pages"])} for c in chunked],
        ids=[f"chunk_{i}" for i in range(len(chunked))],
    )

    logger.info("[%s] Knowledge base created with %d chunks", request_id, len(chunked))
    return collection


def _hybrid_search_bm25(chunks: list[dict], query: str, top_k: int = 10) -> list[tuple[int, float]]:
    """Run BM25 keyword search on chunks and return top-k indices with scores."""
    if not BM25_AVAILABLE or not chunks:
        return []
    
    corpus = [c["text"] for c in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, scores[i]) for i in top_indices]


def _rerank_with_cohere(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """Rerank documents using CoHERE API. Returns list of (index, score) tuples."""
    if not COHERE_AVAILABLE:
        return [(i, 1.0) for i in range(min(len(documents), top_n))]
    
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        logger.warning("COHERE_API_KEY not set, skipping reranking")
        return [(i, 1.0) for i in range(min(len(documents), top_n))]
    
    try:
        client = cohere.Client(cohere_api_key)
        response = client.rerank(
            query=query,
            documents=documents[:20],
            top_n=top_n,
            model="rerank-multilingual-v3.0"
        )
        return [(r.index, r.relevance_score) for r in response.results]
    except Exception as e:
        logger.warning(f"CoHERE reranking failed: {e}")
        return [(i, 1.0) for i in range(min(len(documents), top_n))]


def search_knowledge(collection, query, n_results=5, use_reranker: bool = False, use_hybrid: bool = False):
    """Search the knowledge base for information relevant to a query.

    Args:
        collection: ChromaDB collection to search.
        query: The user's question.
        n_results: Number of top results to return.
        use_reranker: Whether to apply CoHERE reranking.
        use_hybrid: Whether to combine BM25 + semantic search.

    Returns:
        Tuple of (context_text, page_numbers) where page_numbers is a sorted
        list of unique page numbers across all matching chunks.
    """
    request_id = str(uuid.uuid4())[:8]
    
    if use_hybrid and BM25_AVAILABLE:
        all_docs = collection.get()
        if all_docs.get("documents"):
            bm25_results = _hybrid_search_bm25(
                [{"text": d} for d in all_docs["documents"]], query, top_k=n_results * 2
            )
            bm25_indices = [idx for idx, _ in bm25_results[:n_results]]
            
            semantic_results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )
            
            semantic_indices = set()
            if semantic_results.get("ids") and semantic_results["ids"][0]:
                for id_str in semantic_results["ids"][0]:
                    doc_idx = int(id_str.split("_")[1]) if "_" in id_str else 0
                    semantic_indices.add(doc_idx)
            
            combined_indices = list(set(bm25_indices) | semantic_indices)[:n_results]
            
            combined_docs = []
            for idx in combined_indices:
                if idx < len(all_docs["documents"]):
                    combined_docs.append(all_docs["documents"][idx])
            
            results_documents = combined_docs[:n_results]
            results_metadatas = []
            for idx in combined_indices[:n_results]:
                if idx < len(all_docs.get("metadatas", [])):
                    results_metadatas.append(all_docs["metadatas"][idx])
        else:
            results_documents = []
            results_metadatas = []
    else:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        results_documents = results["documents"][0] if results.get("documents") and results["documents"][0] else []
        results_metadatas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else []

    if use_reranker and COHERE_AVAILABLE and results_documents:
        reranked = _rerank_with_cohere(query, results_documents, top_n=n_results)
        reranked_docs = [results_documents[idx] for idx, _ in reranked]
        reranked_metas = [results_metadatas[idx] for idx, _ in reranked]
        results_documents = reranked_docs
        results_metadatas = reranked_metas
        logger.info("[%s] Reranked results using CoHERE", request_id)

    if results_documents:
        context = "\n\n".join(results_documents)

        page_set = set()
        for meta in results_metadatas:
            pages_str = meta.get("pages", "")
            if pages_str:
                for p in pages_str.split(","):
                    p = p.strip()
                    if p:
                        page_set.add(int(p))

        pages = sorted(page_set)
        logger.info(
            "[%s] Found %d relevant chunks (pages %s) for query: '%s'",
            request_id, len(results_documents), pages or "N/A", query[:80],
        )
        return context, pages

    logger.warning("[%s] No relevant information found for query: '%s'", request_id, query[:80])
    return "No relevant information found.", []


def reset_knowledge_base():
    """Delete the persistent knowledge base to start fresh."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Knowledge base reset successfully")
    except Exception:
        pass