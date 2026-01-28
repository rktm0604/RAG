import gradio as gr
import ollama
from pdf_reader import load_pdf, create_knowledge_base, search_knowledge

# Global variable to store knowledge base
knowledge_base = None

def upload_pdfs(files):
    """Process uploaded PDF files"""
    global knowledge_base
    
    if not files:
        return "❌ No files uploaded!"
    
    try:
        all_text = ""
        file_names = []
        
        for file in files:
            print(f"Processing: {file.name}")
            text = load_pdf(file.name)
            all_text += text + "\n\n"
            file_names.append(file.name.split("\\")[-1])
        
        print("Creating knowledge base...")
        knowledge_base = create_knowledge_base(all_text)
        
        return f"✅ Successfully loaded {len(files)} PDF(s): {', '.join(file_names)}\n\nYou can now ask questions!"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_with_context(message, history):
    """Chat function that uses PDF context"""
    global knowledge_base
    
    if knowledge_base is None:
        return "⚠️ Please upload PDF files first using the upload section above!"
    
    try:
        # Search for relevant context from PDFs
        context = search_knowledge(knowledge_base, message)
        
        # Create prompt with context
        prompt = f"""Based on the following information from the uploaded documents:

{context}

Question: {message}

Please answer the question using the information provided above. If the information doesn't contain the answer, say so."""
        
        # Get AI response
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create the interface
with gr.Blocks(title="RAG Study Assistant") as demo:
    gr.Markdown("# 📚 RAG Study Assistant")
    gr.Markdown("Upload your study materials (PDFs) and ask questions about them!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload PDFs")
            file_upload = gr.File(
                file_count="multiple",
                label="Select PDF files",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Process PDFs", variant="primary")
            status = gr.Textbox(label="Status", lines=3)
    
    gr.Markdown("---")
    gr.Markdown("### 💬 Ask Questions")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_context,
        examples=[
            "What are the main topics covered?",
            "Explain the key concepts",
            "Summarize the important points"
        ]
    )
    
    # Connect upload button
    upload_btn.click(
        fn=upload_pdfs,
        inputs=[file_upload],
        outputs=[status]
    )

print("Starting RAG Study Assistant...")
demo.launch()