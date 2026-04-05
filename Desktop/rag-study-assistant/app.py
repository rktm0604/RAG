"""RAG Study Assistant — Gradio web application."""

import logging
import os
import time
from pathlib import Path

import gradio as gr
import ollama

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Will use system env vars instead

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
# Core functions
# ---------------------------------------------------------------------------

def upload_pdfs(files, state):
    """Process uploaded PDF files and build a knowledge base."""
    if not files:
        return "⚠️ **Please upload at least one PDF file!**", "*No documents loaded yet*", state

    try:
        all_pages = []
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

        status = f"✅ **Successfully loaded {len(files)} PDF(s)**\n\n"
        for info in uploaded_files_info:
            status += f"📄 **{info['name']}**\n"
            status += f"   - Characters: {info['chars']:,}\n"
            status += f"   - Pages: {info['pages']}\n\n"
        status += "💡 You can now ask questions about your documents!"

        files_list = "### 📚 Loaded Documents:\n\n"
        for info in uploaded_files_info:
            files_list += f"- **{info['name']}** ({info['pages']} pages, {info['chars']:,} chars)\n"

        state["knowledge_base"] = collection
        state["files"] = uploaded_files_info
        state["conversation"] = []

        return status, files_list, state

    except Exception as e:
        logger.error("Error uploading PDFs: %s", e, exc_info=True)
        return f"❌ **Error processing PDFs:**\n\n{str(e)}", "*No documents loaded yet*", state


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
        context, source_pages = search_knowledge(kb, message)

        history_text = ""
        recent = history[-3:] if len(history) > 3 else history
        if recent:
            history_text = "\n📝 **Recent conversation:**\n"
            for entry in recent:
                if isinstance(entry, dict):
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    history_text += f"**{'You' if role == 'user' else 'Assistant'}:** {content}\n"
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    history_text += f"**You:** {entry[0]}\n**Assistant:** {entry[1]}\n"

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

        footer = "\n\n---\n\n📚 *Answer generated from your uploaded documents*"
        if source_pages:
            page_str = ", ".join(str(p) for p in source_pages)
            footer += f"  ·  📄 *Source pages: {page_str}*"
        yield response_text + footer

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
        yield f"❌ **Error generating response:**\n\n{str(e)}\n\n---\n\n**Troubleshooting:**\n1. Make sure Ollama is running (`ollama serve`)\n2. Verify the model is installed (`ollama list`)\n3. Try running `ollama pull {OLLAMA_MODEL}`"


def clear_all(state):
    """Reset everything and start fresh."""
    try:
        reset_knowledge_base()
    except Exception as e:
        logger.warning(f"Error resetting knowledge base: {e}")
    
    state = {"knowledge_base": None, "files": [], "conversation": []}
    logger.info("Session cleared")
    return (
        "🔄 **All data cleared.**\n\nUpload new PDFs to start fresh!",
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
    export += f"📅 Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"

    for i, entry in enumerate(conversation, 1):
        export += f"## 💬 Q{i}: {entry['question']}\n\n"
        export += f"**Answer:**\n{entry['answer']}\n\n"
        if entry.get("pages"):
            export += f"📄 *Source pages: {', '.join(str(p) for p in entry['pages'])}*\n"
        export += f"*🕐 {entry['timestamp']}*\n\n---\n\n"

    return gr.update(value=export, visible=True)


# ---------------------------------------------------------------------------
# Enhanced Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
}
.main-header h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
}
.main-header p {
    font-size: 1.1em;
    opacity: 0.9;
}
.status-box {
    padding: 15px;
    border-radius: 8px;
    background: #f8f9fa;
    border-left: 4px solid #667eea;
}
.session-card {
    padding: 15px;
    border-radius: 8px;
    background: #f8f9fa;
}
.tips-card {
    padding: 15px;
    border-radius: 8px;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
}
.upload-section {
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="RAG Study Assistant", css=CSS, theme=gr.themes.Soft()) as demo:
    app_state = gr.State({"knowledge_base": None, "files": [], "conversation": []})

    gr.HTML("""
    <div class="main-header">
        <h1>📚 RAG Study Assistant</h1>
        <p>AI-Powered Document Q&A with RAG Pipeline • Local & Private</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML('<div class="upload-section">')
            gr.Markdown("### 📤 Upload Documents")
            file_upload = gr.File(
                file_count="multiple",
                label="Select PDF files (multiple allowed)",
                file_types=[".pdf"],
            )

            with gr.Row():
                upload_btn = gr.Button("📥 Process PDFs", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="stop")

            status = gr.Markdown("*No documents loaded yet*")
            gr.HTML('</div>')

            gr.Markdown("---")
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.ChatInterface(
                fn=chat_with_context,
                additional_inputs=[app_state],
                examples=[
                    ["What are the main topics in these documents?"],
                    ["Summarize the key points"],
                    ["Explain the most important concepts"],
                    ["What does the document say about...?"],
                ],
                title="",
                description="",
            )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📊 Session Info")
                files_display = gr.Markdown("*No documents loaded yet*")

            gr.Markdown("---")
            
            with gr.Group():
                gr.Markdown("### 💾 Export")
                export_btn = gr.Button("📄 Export Conversation", size="sm")
                export_output = gr.Textbox(label="Conversation History", lines=8, visible=False)

            gr.Markdown("---")
            
            with gr.Group():
                gr.Markdown("""
                ### 💡 Tips
                - 📄 Upload multiple PDFs at once
                - 🎯 Ask specific questions
                - 🔄 Reference previous questions
                - 📥 Export for notes

                ### 🔧 Configuration
                - **Model:** {model}
                - **Embeddings:** bge-small-en-v1.5
                - **Vector DB:** ChromaDB
                """.format(model=OLLAMA_MODEL))

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