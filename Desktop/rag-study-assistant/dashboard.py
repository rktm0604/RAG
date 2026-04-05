import gradio as gr

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

body {
    background: #f8f9fa;
}

/* Navbar */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 15px 40px;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    position: sticky;
    top: 0;
    z-index: 100;
}

.navbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo {
    font-size: 24px;
    font-weight: 700;
    color: #667eea;
}

.logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
}

.navbar-center {
    display: flex;
    gap: 30px;
}

.nav-item {
    font-size: 15px;
    font-weight: 500;
    color: #555;
    cursor: pointer;
    transition: color 0.2s;
}

.nav-item:hover {
    color: #667eea;
}

.nav-item.active {
    color: #667eea;
    font-weight: 600;
}

.navbar-right {
    display: flex;
    align-items: center;
    gap: 20px;
}

.search-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 20px;
    color: #888;
}

.get-started-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 25px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}

.get-started-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}

/* Hero Section */
.hero {
    padding: 60px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
    animation: float 20s infinite;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    right: -10%;
    width: 50%;
    height: 150%;
    background: radial-gradient(circle, rgba(118, 75, 162, 0.08) 0%, transparent 70%);
    animation: float 25s infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(20px, 20px) rotate(5deg); }
}

.hero h1 {
    font-size: 48px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}

.hero p {
    font-size: 18px;
    color: #666;
    position: relative;
    z-index: 1;
}

/* Cards Section */
.cards-section {
    padding: 40px;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    color: #1a1a2e;
    text-align: center;
    margin-bottom: 40px;
}

.cards-container {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
}

.course-card {
    width: 320px;
    border-radius: 20px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.3s, box-shadow 0.3s;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    background: white;
}

.course-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.15);
}

.card-image {
    height: 180px;
    background-size: cover;
    background-position: center;
    position: relative;
}

.card-tag {
    position: absolute;
    top: 15px;
    left: 15px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.tag-green {
    background: #d4edda;
    color: #155724;
}

.tag-orange {
    background: #fff3cd;
    color: #856404;
}

.tag-blue {
    background: #d1ecf1;
    color: #0c5460;
}

.card-content {
    padding: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 8px;
}

.card-subtitle {
    font-size: 14px;
    color: #888;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Green Card */
.card-green .card-image {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
}

.card-green .card-icon {
    background: #28a745;
}

/* Orange Card */
.card-orange .card-image {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
}

.card-orange .card-icon {
    background: #fd7e14;
}

/* Blue Card */
.card-blue .card-image {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
}

.card-blue .card-icon {
    background: #17a2b8;
}

.card-icon {
    width: 50px;
    height: 50px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: white;
    position: absolute;
    bottom: -25px;
    left: 20px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

/* Details Section */
.details-section {
    padding: 30px 40px;
    background: white;
    margin: 20px 40px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    display: none;
}

.details-section.visible {
    display: block;
    animation: fadeIn 0.5s;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.details-content {
    display: flex;
    gap: 40px;
    align-items: start;
}

.details-image {
    width: 300px;
    height: 200px;
    border-radius: 15px;
    background: #f8f9fa;
}

.details-info h2 {
    font-size: 24px;
    color: #1a1a2e;
    margin-bottom: 10px;
}

.details-info p {
    color: #666;
    line-height: 1.6;
}

.details-meta {
    display: flex;
    gap: 20px;
    margin-top: 20px;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #888;
    font-size: 14px;
}

/* Progress Bar */
.progress-section {
    padding: 20px 40px 60px;
    text-align: center;
}

.progress-bar {
    width: 100%;
    max-width: 600px;
    height: 8px;
    background: #e9ecef;
    border-radius: 10px;
    overflow: hidden;
    margin: 20px auto;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    animation: progress 2s ease-in-out infinite;
}

@keyframes progress {
    0% { width: 20%; }
    50% { width: 80%; }
    100% { width: 20%; }
}

/* Responsive */
@media (max-width: 768px) {
    .navbar {
        padding: 15px 20px;
    }
    .navbar-center {
        display: none;
    }
    .hero h1 {
        font-size: 32px;
    }
    .cards-container {
        flex-direction: column;
        align-items: center;
    }
    .course-card {
        width: 100%;
        max-width: 350px;
    }
    .details-content {
        flex-direction: column;
    }
}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    
    selected_course = gr.State(None)
    
    # Navbar
    with gr.Row():
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
        """)
    
    # Hero Section
    with gr.Row():
        gr.HTML("""
        <div class="hero">
            <h1>📚 RAG Study Assistant</h1>
            <p>Upload your PDFs and get AI-powered answers with citations</p>
        </div>
        """)
    
    # Cards Section
    with gr.Row():
        gr.HTML("""
        <div class="cards-section">
            <h2 class="section-title">🚀 Featured Capabilities</h2>
            <div class="cards-container">
                <div class="course-card card-green" onclick="selectCard('semantic')">
                    <div class="card-image">
                        <span class="card-tag tag-green">Advanced</span>
                        <div class="card-icon">🔍</div>
                    </div>
                    <div class="card-content">
                        <h3 class="card-title">Semantic Search</h3>
                        <p class="card-subtitle">🕐 State-of-the-art embeddings</p>
                    </div>
                </div>
                <div class="course-card card-orange" onclick="selectCard('hybrid')">
                    <div class="card-image">
                        <span class="card-tag tag-orange">Pro</span>
                        <div class="card-icon">⚡</div>
                    </div>
                    <div class="card-content">
                        <h3 class="card-title">Hybrid Search</h3>
                        <p class="card-subtitle">🕐 BM25 + Vector combined</p>
                    </div>
                </div>
                <div class="course-card card-blue" onclick="selectCard('rerank')">
                    <div class="card-image">
                        <span class="card-tag tag-blue">Beta</span>
                        <div class="card-icon">🎯</div>
                    </div>
                    <div class="card-content">
                        <h3 class="card-title">AI Reranking</h3>
                        <p class="card-subtitle">🕐 CoHERE powered precision</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Hidden click handlers (simulated via JavaScript)
    with gr.Row():
        gr.HTML("""
        <script>
        function selectCard(type) {
            document.querySelector('.details-section').classList.add('visible');
            if (type === 'semantic') {
                document.getElementById('details-title').innerText = '🔍 Semantic Search with BGE Embeddings';
                document.getElementById('details-desc').innerText = 'State-of-the-art BAAI/bge-small-en-v1.5 embeddings provide superior semantic understanding. Unlike basic keyword matching, BGE captures meaning and context, enabling more accurate document retrieval even when queries use different terminology than the source material.';
            } else if (type === 'hybrid') {
                document.getElementById('details-title').innerText = '⚡ Hybrid Search (BM25 + Semantic)';
                document.getElementById('details-desc').innerText = 'Combines the best of both worlds: BM25 keyword matching with semantic vector search. This hybrid approach ensures you find relevant documents even when they dont contain exact keyword matches, while maintaining precision for specific terms.';
            } else if (type === 'rerank') {
                document.getElementById('details-title').innerText = '🎯 AI-Powered Reranking with CoHERE';
                document.getElementById('details-desc').innerText = 'Advanced reranking using Cohere rerank-multilingual-v3.0 improves result quality by reordering initial retrieved documents based on deeper semantic understanding, ensuring the most relevant results appear at the top.';
            }
        }
        </script>
        """)
    
    # Details Section
    with gr.Row():
        gr.HTML("""
        <div class="details-section">
            <div class="details-content">
                <div class="details-image" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; font-size: 60px;">
                    📊
                </div>
                <div class="details-info">
                    <h2 id="details-title">🔍 Semantic Search with BGE Embeddings</h2>
                    <p id="details-desc">State-of-the-art BAAI/bge-small-en-v1.5 embeddings provide superior semantic understanding. Unlike basic keyword matching, BGE captures meaning and context, enabling more accurate document retrieval.</p>
                    <div class="details-meta">
                        <div class="meta-item">📈 High Accuracy</div>
                        <div class="meta-item">⚡ Fast Inference</div>
                        <div class="meta-item">🔒 Private & Local</div>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Progress Section
    with gr.Row():
        gr.HTML("""
        <div class="progress-section">
            <h3 style="color: #666; font-weight: 500;">System Performance</h3>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <p style="color: #888; font-size: 14px;">Embedding model loading • ChromaDB ready • Ollama connected</p>
        </div>
        """)

demo.launch()