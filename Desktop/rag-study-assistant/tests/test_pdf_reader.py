"""Unit tests for pdf_reader module."""

import pytest
from pdf_reader import (
    chunk_text,
    chunk_text_with_pages,
    create_knowledge_base,
    load_pdf,
    pages_to_text,
    reset_knowledge_base,
    search_knowledge,
)

# ───────────────────────── chunk_text ─────────────────────────


class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello, world!"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunks_have_overlap(self):
        # Create text that spans multiple chunks
        text = "A" * 500 + ". " + "B" * 500 + ". " + "C" * 500
        chunks = chunk_text(text, chunk_size=600, chunk_overlap=200)
        assert len(chunks) >= 2

        # Verify overlap: content from the end of chunk N appears at start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-100:]
            head = chunks[i + 1][:300]
            overlap_found = any(tail[j : j + 20] in head for j in range(len(tail) - 20))
            assert overlap_found, f"No overlap found between chunk {i} and chunk {i+1}"

    def test_sentence_boundary_splitting(self):
        text = "First sentence. Second sentence. Third sentence is much longer and goes on and on."
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
        assert len(chunks) >= 2

    def test_no_infinite_loop_on_tiny_chunks(self):
        text = "A very short text."
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=3)
        assert len(chunks) >= 1


# ───────────────── chunk_text_with_pages ─────────────────


class TestChunkTextWithPages:
    def test_empty_pages_returns_empty(self):
        assert chunk_text_with_pages([]) == []

    def test_single_page_metadata(self):
        pages = [(1, "Hello world. " * 20)]
        result = chunk_text_with_pages(pages, chunk_size=50, chunk_overlap=10)
        assert len(result) >= 1
        for item in result:
            assert "text" in item
            assert "pages" in item
            assert 1 in item["pages"]

    def test_multi_page_metadata(self):
        pages = [
            (1, "Page one content. " * 30),
            (2, "Page two content. " * 30),
            (3, "Page three content. " * 30),
        ]
        result = chunk_text_with_pages(pages, chunk_size=100, chunk_overlap=20)
        assert len(result) >= 3

        # Collect all page numbers seen across chunks
        all_pages = set()
        for item in result:
            all_pages.update(item["pages"])

        # All three source pages should appear somewhere
        assert 1 in all_pages
        assert 2 in all_pages
        assert 3 in all_pages

    def test_chunk_spanning_pages(self):
        # Short pages that will be merged into one chunk
        pages = [
            (5, "Short A. "),
            (6, "Short B. "),
        ]
        result = chunk_text_with_pages(pages, chunk_size=1000, chunk_overlap=200)
        assert len(result) == 1
        # The single chunk should reference both pages
        assert 5 in result[0]["pages"]
        assert 6 in result[0]["pages"]


# ──────────────────── load_pdf ────────────────────


class TestLoadPdf:
    def test_load_valid_pdf(self):
        """Test loading the included sample PDF."""
        import os

        pdf_path = os.path.join(os.path.dirname(__file__), "..", "kundu_test.pdf")
        if not os.path.exists(pdf_path):
            pytest.skip("kundu_test.pdf not found")

        pages = load_pdf(pdf_path)
        assert isinstance(pages, list)
        assert len(pages) >= 1

        # Each entry is (page_number, text)
        for page_num, text in pages:
            assert isinstance(page_num, int)
            assert isinstance(text, str)
            assert page_num >= 1

        # Total text should be substantial
        total = sum(len(t) for _, t in pages)
        assert total > 100

    def test_load_nonexistent_pdf_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent_file_12345.pdf")


# ───────────────── pages_to_text ──────────────────


class TestPagesToText:
    def test_basic_conversion(self):
        pages = [(1, "Hello"), (2, "World")]
        result = pages_to_text(pages)
        assert "Hello" in result
        assert "World" in result

    def test_empty_pages(self):
        assert pages_to_text([]) == ""


# ──────────── create & search knowledge base ─────────────


class TestKnowledgeBase:
    def setup_method(self):
        reset_knowledge_base()

    def teardown_method(self):
        reset_knowledge_base()

    def test_create_and_search_with_pages(self):
        pages = [
            (1, "Machine learning is a subset of artificial intelligence. "),
            (2, "Neural networks are inspired by biological neurons. "),
            (3, "Deep learning uses multiple layers of neural networks. "),
            (4, "Supervised learning uses labeled training data. "),
            (5, "Unsupervised learning finds patterns without labels."),
        ]
        collection = create_knowledge_base(pages, chunk_size=100, chunk_overlap=30)
        context, result_pages = search_knowledge(collection, "What is machine learning?")

        assert isinstance(context, str)
        assert len(context) > 0
        assert "machine learning" in context.lower() or "artificial intelligence" in context.lower()

        # Should return page numbers
        assert isinstance(result_pages, list)
        assert len(result_pages) > 0
        assert all(isinstance(p, int) for p in result_pages)

    def test_create_with_plain_string_backward_compat(self):
        """Backward compatibility: plain string still works."""
        text = "Python is a programming language. " * 20
        collection = create_knowledge_base(text, chunk_size=100, chunk_overlap=20)
        context, pages = search_knowledge(collection, "What is Python?")

        assert isinstance(context, str)
        assert isinstance(pages, list)

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            create_knowledge_base("")

    def test_search_returns_tuple(self):
        text = "Python is a programming language. " * 20
        collection = create_knowledge_base(text, chunk_size=100, chunk_overlap=20)
        result = search_knowledge(collection, "What is Python?")

        # Should return (context_str, pages_list)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)
