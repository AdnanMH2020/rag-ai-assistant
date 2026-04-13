from dotenv import load_dotenv
load_dotenv()

import os

from app.rag.loader import load_and_split
from app.rag.retriever import create_vector_store
from app.rag.pipeline import ask_question
file = ["SoftwareText.pdf", "sample.pdf"]
pdf_path = os.path.join("data", file[0])

print("Loading document...")
chunks = load_and_split(pdf_path)

print("Creating vector store...")
vectorstore = create_vector_store(chunks)

print("\nRAG system ready. Type 'exit' to quit.")

while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    try:
        answer = ask_question(vectorstore, query)
        print("\nAnswer:", answer)
    except Exception as e:
        print("\nError:", e)