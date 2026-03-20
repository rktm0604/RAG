"""RAG Study Assistant — Gradio web application."""

import logging
import os
import time
from pathlib import Path

import gradio as gr
import ollama

from pdf_reader import (
    create_knowledge_base,
    load_pdf,
    pages_to_text,
    reset_knowledge_base,
    search_knowledge,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup health-check
# ---------------------------------------------------------------------------

def _check_ollama():
    """Verify Ollama is reachable and the model is available."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, "models") else []
        if not any(OLLAMA_MODEL in m for m in available):
            logger.warning(
                "⚠️  Model '%s' not found in Ollama. Available: %s. "
                "Run: ollama pull %s",
                OLLAMA_MODEL, available, OLLAMA_MODEL,
            )
        else:
            logger.info("✅ Ollama is running, model '%s' is available", OLLAMA_MODEL)
    except Exception as e:
        logger.error(
            "❌ Cannot connect to Ollama: %s — make sure 'ollama serve' is running.", e,
        )

_check_ollama()


# ---------------------------------------------------------------------------
# Core functions (use Gradio State instead of globals)
# ---------------------------------------------------------------------------

def upload_pdfs(files, state):
    """Process uploaded PDF files and build a knowledge base."""
    if not files:
        return "❌ Please upload at least one PDF file!", "*No documents loaded yet*", state

    try:
        all_pages = []          # list of (page_num, text) across all PDFs
        uploaded_files_info = []

        for file in files:
            filepath = Path(file.name)
            logger.info("Processing file: %s", filepath.name)

            pages = load_pdf(str(filepath))
            all_pages.extend(pages)

            total_chars = sum(len(t) for _, t in pages)
            uploaded_files_info.append({
                "name": filepath.name,
                "chars": total_chars,
                "pages": len(pages),
            })

        logger.info("Creating knowledge base from %d file(s)...", len(files))
        collection = create_knowledge_base(all_pages)

        # Build status message
        status = f"✅ **Successfully loaded {len(files)} PDF(s)**\n\n"
        for info in uploaded_files_info:
            status += f"📄 **{info['name']}**\n"
            status += f"   - Characters: {info['chars']:,}\n"
            status += f"   - Pages: {info['pages']}\n\n"
        status += "💡 You can now ask questions about your documents!"

        # Sidebar file list
        files_list = "### 📚 Loaded Documents:\n\n"
        for info in uploaded_files_info:
            files_list += f"- {info['name']} ({info['pages']} pages)\n"

        # Update state
        state["knowledge_base"] = collection
        state["files"] = uploaded_files_info
        state["conversation"] = []

        return status, files_list, state

    except Exception as e:
        logger.error("Error uploading PDFs: %s", e, exc_info=True)
        return f"❌ Error processing PDFs: {e}", "*No documents loaded yet*", state


def chat_with_context(message, history, state):
    """Answer a user question using RAG context and conversation memory."""
    kb = state.get("knowledge_base")

    if kb is None:
        yield "⚠️ **Please upload PDF files first!**\n\nUse the upload section above to add your study materials."
        return

    if not message.strip():
        yield ""
        return

    try:
        # Retrieve relevant context + page numbers from the knowledge base
        context, source_pages = search_knowledge(kb, message)

        # Build recent conversation context
        history_text = ""
        recent = history[-3:] if len(history) > 3 else history
        if recent:
            history_text = "\nRecent conversation:\n"
            for entry in recent:
                if isinstance(entry, dict):
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    history_text += f"{'User' if role == 'user' else 'Assistant'}: {content}\n"
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    history_text += f"User: {entry[0]}\nAssistant: {entry[1]}\n"

        prompt = f"""You are a helpful study assistant. Answer the question based on the provided context from uploaded documents.

Context from documents:
{context}
{history_text}

Current question: {message}

Instructions:
1. Answer based primarily on the provided context
2. Be specific and cite information from the context
3. If the context doesn't contain the answer, say so clearly
4. Keep answers concise but complete
5. Use the conversation history for context if relevant

Answer:"""

        # Stream response from Ollama
        response_text = ""
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                response_text += chunk["message"]["content"]
                yield response_text

        # Append source citation footer
        footer = "\n\n---\n\n📚 *Answer generated from your uploaded documents*"
        if source_pages:
            page_str = ", ".join(str(p) for p in source_pages)
            footer += f"  ·  📄 *Source pages: {page_str}*"
        yield response_text + footer

        # Track conversation
        conversation = state.get("conversation", [])
        conversation.append({
            "question": message,
            "answer": response_text,
            "pages": source_pages,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["conversation"] = conversation

    except Exception as e:
        logger.error("Error generating response: %s", e, exc_info=True)
        yield (
            f"❌ **Error generating response:** {e}\n\n"
            "Please check that:\n"
            f"1. Ollama is running (`ollama serve`)\n"
            f"2. The model is installed (`ollama pull {OLLAMA_MODEL}`)"
        )


def clear_all(state):
    """Reset everything and start fresh."""
    reset_knowledge_base()
    state = {"knowledge_base": None, "files": [], "conversation": []}
    logger.info("Session cleared")
    return (
        "🔄 All data cleared. Upload new PDFs to start fresh.",
        "*No documents loaded yet*",
        None,
        state,
    )


def export_conversation(state):
    """Export the conversation history as markdown."""
    conversation = state.get("conversation", [])

    if not conversation:
        return gr.update(value="No conversation to export yet.", visible=True)

    export = "# RAG Study Assistant - Conversation History\n\n"
    export += f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

    for i, entry in enumerate(conversation, 1):
        export += f"## Q{i}: {entry['question']}\n\n"
        export += f"**Answer:** {entry['answer']}\n\n"
        if entry.get("pages"):
            export += f"📄 *Source pages: {', '.join(str(p) for p in entry['pages'])}*\n\n"
        export += f"*Time: {entry['timestamp']}*\n\n---\n\n"

    return gr.update(value=export, visible=True)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG Study Assistant") as demo:
    # Shared state across all components
    app_state = gr.State({"knowledge_base": None, "files": [], "conversation": []})

    gr.Markdown("""
    # 📚 RAG Study Assistant
    ### AI-Powered Document Q&A with Conversation Memory

    Upload your study materials and ask questions. The AI will find relevant information and provide detailed answers.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Upload section
            with gr.Group():
                gr.Markdown("### 📤 Upload Documents")
                file_upload = gr.File(
                    file_count="multiple",
                    label="Select PDF files (you can upload multiple)",
                    file_types=[".pdf"],
                )

                with gr.Row():
                    upload_btn = gr.Button("📥 Process PDFs", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear All", variant="stop")

                status = gr.Markdown(label="Status")

            # Chat section
            gr.Markdown("---")
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.ChatInterface(
                fn=chat_with_context,
                additional_inputs=[app_state],
                examples=[
                    ["What are the main topics covered in these documents?"],
                    ["Can you summarize the key points?"],
                    ["Explain the most important concepts from the materials"],
                    ["What does the document say about the introduction?"],
                ],
            )

        with gr.Column(scale=1):
            # Sidebar
            gr.Markdown("### 📊 Session Info")
            files_display = gr.Markdown("*No documents loaded yet*")

            gr.Markdown("---")
            gr.Markdown("### 💾 Export")
            export_btn = gr.Button("📄 Export Conversation", size="sm")
            export_output = gr.Textbox(label="Conversation History", lines=10, visible=False)

            gr.Markdown("---")
            gr.Markdown(f"""
            ### 💡 Tips
            - Upload multiple PDFs at once
            - Ask specific questions for better answers
            - Reference previous questions in follow-ups
            - Export your conversation for notes

            ### 🔧 Tech Stack
            - **AI Model:** {OLLAMA_MODEL}
            - **Vector DB:** ChromaDB
            - **Interface:** Gradio
            - **Runs on:** Local GPU
            """)

    # Wire up event handlers
    upload_btn.click(
        fn=upload_pdfs,
        inputs=[file_upload, app_state],
        outputs=[status, files_display, app_state],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[app_state],
        outputs=[status, files_display, file_upload, app_state],
    )

    export_btn.click(
        fn=export_conversation,
        inputs=[app_state],
        outputs=[export_output],
    )

logger.info("🚀 Starting RAG Study Assistant (model: %s)...", OLLAMA_MODEL)
logger.info("📍 Open browser at: http://localhost:7860")
demo.launch()