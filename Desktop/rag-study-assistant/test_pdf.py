from pdf_reader import load_pdf, create_knowledge_base, search_knowledge

# Test with your PDF
pdf_path = input("Enter full path to your PDF: ")

print("Loading PDF...")
text = load_pdf(pdf_path)
print(f"Loaded {len(text)} characters")

print("\nCreating knowledge base...")
kb = create_knowledge_base(text)

print("\nTesting search...")
query = input("Ask a question about the PDF: ")
context = search_knowledge(kb, query)
print("\nRelevant context found:")
print(context)