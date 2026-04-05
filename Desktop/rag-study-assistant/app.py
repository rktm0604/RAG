import gradio as gr
import logging
import os
import time
from pathlib import Path

import ollama

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pdf_reader import (
    create_knowledge_base,
    load_pdf,
    reset_knowledge_base,
    search_knowledge,
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_ollama():
    try:
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, "models") else []
        if not any(OLLAMA_MODEL in m for m in available):
            logger.warning("Model '%s' not found in Ollama. Run: ollama pull %s", OLLAMA_MODEL, OLLAMA_MODEL)
        else:
            logger.info("✅ Ollama running with model '%s'", OLLAMA_MODEL)
    except Exception as e:
        logger.error("Cannot connect to Ollama: %s", e)

_check_ollama()


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
body { background: #f8f9fa; }

/* Navbar */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 15px 40px; background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100;
}
.navbar-left { display: flex; align-items: center; gap: 10px; }
.logo { font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-icon {
    width: 36px; height: 36px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    color: white; font-size: 18px;
}
.navbar-center { display: flex; gap: 30px; }
.nav-item { font-size: 15px; font-weight: 500; color: #555; cursor: pointer; transition: color 0.2s; }
.nav-item:hover, .nav-item.active { color: #667eea; font-weight: 600; }
.navbar-right { display: flex; align-items: center; gap: 15px; }
.search-btn { background: none; border: none; cursor: pointer; font-size: 20px; color: #888; }
.get-started-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; padding: 10px 24px;
    border-radius: 25px; font-weight: 500; cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}
.get-started-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }

/* Hero */
.hero {
    padding: 50px 40px; text-align: center;
    background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; left: -20%; width: 60%; height: 200%;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
}
.hero h1 { font-size: 42px; font-weight: 700; color: #1a1a2e; margin-bottom: 10px; position: relative; z-index: 1; }
.hero p { font-size: 16px; color: #666; position: relative; z-index: 1; }

/* Main Content */
.main-content { padding: 30px 40px; display: flex; gap: 30px; }

/* Left Panel - Upload */
.upload-panel { flex: 1; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
.panel-title { font-size: 18px; font-weight: 600; color: #1a1a2e; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.upload-area { border: 2px dashed #ddd; border-radius: 15px; padding: 30px; text-align: center; cursor: pointer; transition: all 0.3s; }
.upload-area:hover { border-color: #667eea; background: #f8f9ff; }
.upload-icon { font-size: 40px; margin-bottom: 10px; }
.btn-row { display: flex; gap: 10px; margin-top: 20px; }
.btn {
    flex: 1; padding: 12px; border-radius: 12px; font-weight: 500; cursor: pointer;
    transition: all 0.2s; border: none;
}
.btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
.btn-secondary { background: #f0f0f0; color: #555; }
.btn-secondary:hover { background: #e0e0e0; }

/* Right Panel - Chat */
.chat-panel { flex: 2; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
.chatbot-container { height: 400px; border: 1px solid #eee; border-radius: 15px; padding: 15px; overflow-y: auto; }
.msg { padding: 12px 16px; border-radius: 12px; margin-bottom: 10px; max-width: 80%; }
.msg-user { background: #667eea; color: white; margin-left: auto; border-bottom-right-radius: 4px; }
.msg-assistant { background: #f0f0f0; color: #333; border-bottom-left-radius: 4px; }
.chat-input { display: flex; gap: 10px; margin-top: 15px; }
.chat-input input {
    flex: 1; padding: 14px 18px; border: 2px solid #eee; border-radius: 25px;
    font-size: 15px; transition: border-color 0.2s;
}
.chat-input input:focus { outline: none; border-color: #667eea; }
.send-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 24px; border-radius: 25px; cursor: pointer; }

/* Status */
.status-box { margin-top: 20px; padding: 15px; border-radius: 12px; background: #f8f9fa; border-left: 4px solid #667eea; }
.status-success { border-left-color: #28a745; background: #d4edda; }
.status-error { border-left-color: #dc3545; background: #f8d7da; }

/* Cards */
.cards-section { padding: 30px 40px; }
.cards-row { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
.feature-card {
    width: 280px; border-radius: 18px; overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08); background: white;
    cursor: pointer; transition: transform 0.3s, box-shadow 0.3s;
}
.feature-card:hover { transform: translateY(-8px); box-shadow: 0 20px 50px rgba(0,0,0,0.15); }
.card-img { height: 120px; display: flex; align-items: center; justify-content: center; font-size: 50px; }
.card-green .card-img { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
.card-orange .card-img { background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); }
.card-blue .card-img { background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); }
.card-body { padding: 18px; }
.card-body h4 { font-size: 17px; color: #1a1a2e; margin-bottom: 5px; }
.card-body p { font-size: 13px; color: #888; }
.card-tag { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-bottom: 8px; }
.tag-green { background: #d4edda; color: #155724; }
.tag-orange { background: #fff3cd; color: #856404; }
.tag-blue { background: #d1ecf1; color: #0c5460; }

/* Footer */
.footer { padding: 30px 40px; text-align: center; color: #888; font-size: 14px; }
.footer a { color: #667eea; text-decoration: none; }

/* Responsive */
@media (max-width: 900px) {
    .main-content { flex-direction: column; }
    .navbar { padding: 15px 20px; }
    .navbar-center { display: none; }
    .hero h1 { font-size: 28px; }
    .feature-card { width: 100%; max-width: 300px; }
}
"""

# State management
app_state = gr.State({"knowledge_base": None, "files": [], "conversation": []})


def upload_pdfs(files, state):
    if not files:
        return "⚠️ Please upload at least one PDF file!", "*No documents loaded yet*", state
    
    try:
        all_pages = []
        uploaded_files_info = []
        
        for file in files:
            filepath = Path(file.name)
            pages = load_pdf(str(filepath))
            all_pages.extend(pages)
            total_chars = sum(len(t) for _, t in pages)
            uploaded_files_info.append({
                "name": filepath.name,
                "chars": total_chars,
                "pages": len(pages),
            })
        
        collection = create_knowledge_base(all_pages)
        
        status = f"✅ **Successfully loaded {len(files)} PDF(s)**\n\n"
        for info in uploaded_files_info:
            status += f"📄 **{info['name']}** — {info['pages']} pages, {info['chars']:,} chars\n"
        status += "\n💡 You can now ask questions!"
        
        state["knowledge_base"] = collection
        state["files"] = uploaded_files_info
        state["conversation"] = []
        
        return status, state
    except Exception as e:
        logger.error("Error uploading PDFs: %s", e, exc_info=True)
        return f"❌ Error: {str(e)}", state


def chat_fn(message, history, state):
    kb = state.get("knowledge_base")
    
    if kb is None:
        yield "⚠️ Please upload PDF files first!"
        return
    
    if not message.strip():
        yield ""
        return
    
    try:
        context, source_pages = search_knowledge(kb, message)
        
        history_text = ""
        recent = history[-3:] if len(history) > 3 else history
        if recent:
            for entry in recent:
                if isinstance(entry, dict):
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    history_text += f"{'You' if role == 'user' else 'Assistant'}: {content}\n"
        
        prompt = f"""You are a helpful study assistant. Answer based on the context.

Context: {context}
{history_text}
Question: {message}

Answer:"""
        
        response_text = ""
        stream = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
        
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                response_text += chunk["message"]["content"]
                yield response_text
        
        footer = "\n\n---\n📚 *Source pages: " + (", ".join(str(p) for p in source_pages) if source_pages else "N/A") + "*"
        yield response_text + footer
        
        state["conversation"].append({"question": message, "answer": response_text, "pages": source_pages})
    except Exception as e:
        logger.error("Error generating response: %s", e)
        yield f"❌ Error: {str(e)}\n\nCheck Ollama is running."


def clear_fn(state):
    try:
        reset_knowledge_base()
    except:
        pass
    state = {"knowledge_base": None, "files": [], "conversation": []}
    return "🔄 All cleared! Upload new PDFs to start.", state


with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="navbar">
        <div class="navbar-left">
            <div class="logo-icon">📚</div>
            <div class="logo">RAG Assistant</div>
        </div>
        <div class="navbar-center">
            <div class="nav-item active">Home</div>
            <div class="nav-item">Documents</div>
            <div class="nav-item">Analytics</div>
            <div class="nav-item">Settings</div>
        </div>
        <div class="navbar-right">
            <button class="search-btn">🔍</button>
            <button class="get-started-btn">Get Started</button>
        </div>
    </div>
    
    <div class="hero">
        <h1>📚 RAG Study Assistant</h1>
        <p>Upload PDFs → Ask Questions → Get AI Answers with Page Citations</p>
    </div>
    """)
    
    with gr.Row():
        gr.HTML("""
        <div class="cards-section">
            <div class="cards-row">
                <div class="feature-card card-green">
                    <div class="card-img">🔍</div>
                    <div class="card-body">
                        <span class="card-tag tag-green">Advanced</span>
                        <h4>Semantic Search</h4>
                        <p>BGE embeddings for superior accuracy</p>
                    </div>
                </div>
                <div class="feature-card card-orange">
                    <div class="card-img">⚡</div>
                    <div class="card-body">
                        <span class="card-tag tag-orange">Pro</span>
                        <h4>Hybrid Search</h4>
                        <p>BM25 + Vector combined</p>
                    </div>
                </div>
                <div class="feature-card card-blue">
                    <div class="card-img">🎯</div>
                    <div class="card-body">
                        <span class="card-tag tag-blue">Beta</span>
                        <h4>AI Reranking</h4>
                        <p>CoHERE powered precision</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    with gr.Row():
        gr.HTML('<div class="main-content">')
        
        with gr.Column():
            gr.HTML('<div class="upload-panel">')
            gr.HTML('<div class="panel-title">📤 Upload Documents</div>')
            file_upload = gr.File(file_count="multiple", label="Select PDF files", file_types=[".pdf"])
            with gr.Row():
                upload_btn = gr.Button("📥 Process", variant="primary", size="sm")
                clear_btn = gr.Button("🗑️ Clear", size="sm")
            status_md = gr.Markdown("*No documents loaded*")
            gr.HTML('</div>')
        
        with gr.Column():
            gr.HTML('<div class="chat-panel">')
            gr.HTML('<div class="panel-title">💬 Ask Questions</div>')
            chatbot = gr.ChatInterface(fn=chat_fn, additional_inputs=[app_state], show_examples=True)
            gr.HTML('</div>')
        
        gr.HTML('</div>')
    
    gr.HTML(f'''
    <div class="footer">
        <p>🤖 Powered by <b>{OLLAMA_MODEL}</b> • 📊 Vector DB: ChromaDB • 🔒 100% Local & Private</p>
    </div>
    ''')
    
    upload_btn.click(fn=upload_pdfs, inputs=[file_upload, app_state], outputs=[status_md, app_state])
    clear_btn.click(fn=clear_fn, inputs=[app_state], outputs=[status_md, app_state])

logger.info("Starting RAG Study Assistant Dashboard...")
demo.launch()