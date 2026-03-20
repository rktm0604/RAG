"""RAG pipeline — PDF loading, chunking with page metadata, and vector search."""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# Optional OCR dependencies (graceful fallback)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ChromaDB persistent storage path
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "study_docs"


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
    # Accept plain string for backward compat (e.g. tests)
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

    # Delete existing collection for a fresh start
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Collection doesn't exist yet — that's fine
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(),
    )

    collection.add(
        documents=[c["text"] for c in chunked],
        metadatas=[{"pages": ",".join(str(p) for p in c["pages"])} for c in chunked],
        ids=[f"chunk_{i}" for i in range(len(chunked))],
    )

    logger.info("Knowledge base created with %d chunks", len(chunked))
    return collection


def search_knowledge(collection, query, n_results=3):
    """Search the knowledge base for information relevant to a query.

    Args:
        collection: ChromaDB collection to search.
        query: The user's question.
        n_results: Number of top results to return.

    Returns:
        Tuple of (context_text, page_numbers) where page_numbers is a sorted
        list of unique page numbers across all matching chunks.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    if results["documents"] and results["documents"][0]:
        context = "\n\n".join(results["documents"][0])

        # Collect page numbers from metadata
        page_set = set()
        if results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                pages_str = meta.get("pages", "")
                if pages_str:
                    for p in pages_str.split(","):
                        p = p.strip()
                        if p:
                            page_set.add(int(p))

        pages = sorted(page_set)
        logger.info(
            "Found %d relevant chunks (pages %s) for query: '%s'",
            len(results["documents"][0]),
            pages or "N/A",
            query[:80],
        )
        return context, pages

    logger.warning("No relevant information found for query: '%s'", query[:80])
    return "No relevant information found.", []


def reset_knowledge_base():
    """Delete the persistent knowledge base to start fresh."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Knowledge base reset successfully")
    except Exception:
        pass