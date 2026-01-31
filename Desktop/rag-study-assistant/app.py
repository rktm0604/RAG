import gradio as gr
import ollama
from pdf_reader import load_pdf, create_knowledge_base, search_knowledge
import time

# Global variables
knowledge_base = None
uploaded_files_info = []
conversation_history = []

def upload_pdfs(files):
    """Process uploaded PDF files"""
    global knowledge_base, uploaded_files_info
    
    if not files:
        return "❌ No files uploaded!", ""
    
    try:
        all_text = ""
        file_names = []
        uploaded_files_info = []
        
        for file in files:
            print(f"Processing: {file.name}")
            text = load_pdf(file.name)
            all_text += text + "\n\n"
            
            filename = file.name.split("\\")[-1].split("/")[-1]
            file_names.append(filename)
            uploaded_files_info.append({
                'name': filename,
                'size': len(text),
                'pages': text.count('\f') + 1 if '\f' in text else 'N/A'
            })
        
        print("Creating knowledge base...")
        knowledge_base = create_knowledge_base(all_text)
        
        # Create detailed status message
        status_msg = f"✅ **Successfully loaded {len(files)} PDF(s)**\n\n"
        
        for info in uploaded_files_info:
            status_msg += f"📄 **{info['name']}**\n"
            status_msg += f"   - Characters: {info['size']:,}\n"
            if info['pages'] != 'N/A':
                status_msg += f"   - Pages: {info['pages']}\n"
            status_msg += "\n"
        
        status_msg += "💡 You can now ask questions about your documents!"
        
        # Show file list in sidebar
        files_list = "### 📚 Loaded Documents:\n\n"
        for info in uploaded_files_info:
            files_list += f"- {info['name']}\n"
        
        return status_msg, files_list
    
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def chat_with_context(message, history):
    """Enhanced chat function with context and conversation memory"""
    global knowledge_base, conversation_history
    
    if knowledge_base is None:
        return "⚠️ **Please upload PDF files first!**\n\nUse the upload section above to add your study materials."
    
    if not message.strip():
        return ""
    
    try:
        # Search for relevant context from PDFs
        context = search_knowledge(knowledge_base, message)
        
        # Build conversation history for better context
        recent_history = history[-3:] if len(history) > 3 else history  # Last 3 exchanges
        
        history_text = ""
        if recent_history:
            history_text = "\nRecent conversation:\n"
            for human, assistant in recent_history:
                history_text += f"User: {human}\nAssistant: {assistant}\n"
        
        # Create enhanced prompt
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
        
        # Get AI response with streaming
        response_text = ""
        stream = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                response_text += chunk['message']['content']
        
        # Add source indicator
        response_with_source = response_text + "\n\n---\n\n📚 *Answer generated from your uploaded documents*"
        
        # Save to conversation history
        conversation_history.append({
            'question': message,
            'answer': response_text,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return response_with_source
    
    except Exception as e:
        return f"❌ **Error generating response:** {str(e)}\n\nPlease try rephrasing your question or check that Ollama is running."

def clear_all():
    """Clear everything and start fresh"""
    global knowledge_base, uploaded_files_info, conversation_history
    knowledge_base = None
    uploaded_files_info = []
    conversation_history = []
    return "🔄 All data cleared. Upload new PDFs to start fresh.", "", None

def export_conversation():
    """Export conversation history"""
    global conversation_history
    
    if not conversation_history:
        return "No conversation to export yet."
    
    export_text = "# RAG Study Assistant - Conversation History\n\n"
    export_text += f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    export_text += "---\n\n"
    
    for i, entry in enumerate(conversation_history, 1):
        export_text += f"## Q{i}: {entry['question']}\n\n"
        export_text += f"**Answer:** {entry['answer']}\n\n"
        export_text += f"*Time: {entry['timestamp']}*\n\n"
        export_text += "---\n\n"
    
    return export_text

# Create enhanced interface
with gr.Blocks(title="RAG Study Assistant - Advanced") as demo:
    gr.Markdown("""
    # 📚 RAG Study Assistant (Advanced)
    ### AI-Powered Document Q&A with Conversation Memory
    
    Upload your study materials and ask questions. The AI will find relevant information and provide detailed answers.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Upload Section
            with gr.Group():
                gr.Markdown("### 📤 Upload Documents")
                file_upload = gr.File(
                    file_count="multiple",
                    label="Select PDF files (you can upload multiple)",
                    file_types=[".pdf"]
                )
                
                with gr.Row():
                    upload_btn = gr.Button("📥 Process PDFs", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear All", variant="stop")
                
                status = gr.Markdown(label="Status")
            
            # Chat Section
            gr.Markdown("---")
            gr.Markdown("### 💬 Ask Questions")
            
            chatbot = gr.ChatInterface(
                fn=chat_with_context,
                examples=[
                    "What are the main topics covered in these documents?",
                    "Can you explain [specific concept] from the materials?",
                    "Summarize the key points about [topic]",
                    "What does the document say about [specific question]?",
                ]
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
            gr.Markdown("""
            ### 💡 Tips
            - Upload multiple PDFs at once
            - Ask specific questions for better answers
            - Reference previous questions in follow-ups
            - Export your conversation for notes
            
            ### 🔧 Tech Stack
            - **AI Model:** Llama 3.2 (3B)
            - **Vector DB:** ChromaDB
            - **Interface:** Gradio
            - **Runs on:** Local GPU
            """)
    
    # Connect buttons
    upload_btn.click(
        fn=upload_pdfs,
        inputs=[file_upload],
        outputs=[status, files_display]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[status, files_display, file_upload]
    )
    
    export_btn.click(
        fn=export_conversation,
        inputs=[],
        outputs=[export_output]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[export_output]
    )

print("🚀 Starting RAG Study Assistant (Advanced Version)...")
print("📍 Open browser at: http://localhost:7860")
demo.launch()