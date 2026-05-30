"""Standalone dashboard demo for RAG Study Assistant."""
import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
body { background: #f8f9fa; }

.navbar { display: flex; align-items: center; justify-content: space-between; padding: 15px 40px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100; }
.navbar-left { display: flex; align-items: center; gap: 10px; }
.logo { font-size: 22px; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; }
.navbar-center { display: flex; gap: 30px; }
.nav-item { font-size: 15px; font-weight: 500; color: #555; cursor: pointer; transition: color 0.2s; }
.nav-item:hover, .nav-item.active { color: #667eea; }
.navbar-right { display: flex; align-items: center; gap: 15px; }

.hero { padding: 60px 40px; text-align: center; position: relative; overflow: hidden; background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%); }
.hero::before { content: ''; position: absolute; top: -50%; left: -20%; width: 60%; height: 200%; background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%); }
.hero h1 { font-size: 42px; font-weight: 700; color: #1a1a2e; margin-bottom: 10px; position: relative; z-index: 1; }
.hero p { font-size: 16px; color: #666; position: relative; z-index: 1; }

.cards-section { padding: 40px; }
.section-title { text-align: center; font-size: 28px; font-weight: 600; color: #1a1a2e; margin-bottom: 40px; }
.cards-container { display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }

.feature-card { width: 300px; border-radius: 20px; overflow: hidden; cursor: pointer; transition: transform 0.3s, box-shadow 0.3s; background: white; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
.feature-card:hover { transform: translateY(-10px); box-shadow: 0 20px 50px rgba(0,0,0,0.15); }
.card-img { height: 140px; display: flex; align-items: center; justify-content: center; font-size: 50px; }
.card-green .card-img { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
.card-orange .card-img { background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); }
.card-blue .card-img { background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); }
.card-body { padding: 20px; }
.card-body h4 { font-size: 18px; font-weight: 600; color: #1a1a2e; margin-bottom: 5px; }
.card-body p { font-size: 13px; color: #888; }
.card-tag { display: inline-block; padding: 5px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.tag-green { background: #d4edda; color: #155724; }
.tag-orange { background: #fff3cd; color: #856404; }
.tag-blue { background: #d1ecf1; color: #0c5460; }

.detail-panel { padding: 30px 40px; background: white; margin: 0 40px 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); min-height: 100px; }
.footer { padding: 40px; text-align: center; color: #888; font-size: 14px; }

@media (max-width: 768px) {
    .navbar-center { display: none; }
    .hero h1 { font-size: 28px; }
    .feature-card { width: 100%; max-width: 350px; }
    .detail-panel { margin: 0 20px 20px; }
}
"""

DETAILS = {
    "semantic": {
        "icon": "🔍",
        "title": "Semantic Search with BGE Embeddings",
        "desc": "State-of-the-art BAAI/bge-small-en-v1.5 embeddings provide superior semantic understanding. Unlike basic keyword matching, BGE captures meaning and context, enabling more accurate document retrieval.",
        "tags": ["📈 High Accuracy", "⚡ Fast Inference", "🔒 Private & Local"],
    },
    "hybrid": {
        "icon": "⚡",
        "title": "Hybrid Search (BM25 + Semantic)",
        "desc": "Combines keyword matching with semantic vector search. Ensures you find relevant documents even without exact keyword matches, while maintaining precision for specific terms.",
        "tags": ["🔀 Combined Approach", "🎯 Precision", "📚 Full Coverage"],
    },
    "rerank": {
        "icon": "🎯",
        "title": "AI-Powered Reranking with CoHERE",
        "desc": "Advanced reranking using Cohere rerank-multilingual-v3.0 improves result quality by reordering retrieved documents based on deeper semantic understanding.",
        "tags": ["🏆 Best Results", "🧠 Deep Understanding", "📊 Improved Ranking"],
    },
}

def show_detail(card_type):
    info = DETAILS[card_type]
    md = f"""
## {info['icon']} {info['title']}

{info['desc']}

<div style="display: flex; gap: 20px; margin-top: 20px;">
    {"".join(f'<div style="background: #f0f0f0; padding: 8px 16px; border-radius: 8px; font-size: 13px;">{t}</div>' for t in info["tags"])}
</div>
"""
    return md


with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="navbar">
        <div class="navbar-left">
            <div class="logo-icon">📚</div>
            <div class="logo">RAG Assistant</div>
        </div>
        <div class="navbar-center">
            <div class="nav-item active">Home</div>
            <div class="nav-item">Features</div>
            <div class="nav-item">About</div>
        </div>
        <div class="navbar-right">
            <button class="get-started-btn" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;padding:10px 24px;border-radius:25px;font-weight:500;cursor:pointer;">Get Started</button>
        </div>
    </div>
    <div class="hero">
        <h1>📚 RAG Study Assistant</h1>
        <p>Upload your PDFs and get AI-powered answers with page citations</p>
    </div>
    <div class="cards-section">
        <h2 class="section-title">🚀 Featured Capabilities</h2>
        <div class="cards-container">
    """)
    
    with gr.Row():
        with gr.Column():
            semantic_btn = gr.Button("", elem_classes="feature-card card-green", min_width=300)
            with gr.Box():
                gr.HTML("""
                <div class="card-img">🔍</div>
                <div class="card-body">
                    <span class="card-tag tag-green">Advanced</span>
                    <h4>Semantic Search</h4>
                    <p>State-of-the-art BGE embeddings</p>
                </div>
                """)
        
        with gr.Column():
            hybrid_btn = gr.Button("", elem_classes="feature-card card-orange", min_width=300)
            with gr.Box():
                gr.HTML("""
                <div class="card-img">⚡</div>
                <div class="card-body">
                    <span class="card-tag tag-orange">Pro</span>
                    <h4>Hybrid Search</h4>
                    <p>BM25 + Vector combined</p>
                </div>
                """)
        
        with gr.Column():
            rerank_btn = gr.Button("", elem_classes="feature-card card-blue", min_width=300)
            with gr.Box():
                gr.HTML("""
                <div class="card-img">🎯</div>
                <div class="card-body">
                    <span class="card-tag tag-blue">Beta</span>
                    <h4>AI Reranking</h4>
                    <p>CoHERE powered precision</p>
                </div>
                """)
    
    gr.HTML("</div></div>")
    
    detail_md = gr.Markdown("", elem_classes="detail-panel")
    
    semantic_btn.click(fn=lambda: show_detail("semantic"), outputs=detail_md)
    hybrid_btn.click(fn=lambda: show_detail("hybrid"), outputs=detail_md)
    rerank_btn.click(fn=lambda: show_detail("rerank"), outputs=detail_md)
    
    gr.HTML("""
    <div class="footer">
        <p>🤖 Powered by Llama 3.2 • 📊 Vector DB: ChromaDB • 🔒 100% Local & Private</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()