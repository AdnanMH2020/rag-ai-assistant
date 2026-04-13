from dotenv import load_dotenv
load_dotenv()

import os

from app.rag.loader import load_and_split
from app.rag.retriever import create_vector_store
from app.rag.pipeline import ask_question


# ----------------------------
# FILE SETUP
# ----------------------------
files = ["SoftwareText.pdf", "sample.pdf"]
pdf_path = os.path.join("data", files[0])

print("Loading document...")
chunks = load_and_split(pdf_path)

print("Creating vector store...")
vectorstore = create_vector_store(chunks)

print("\nRunning system tests...\n")


# =========================================================
# TEST 1: VECTOR STORE CREATION
# =========================================================
def test_vectorstore_exists():
    assert vectorstore is not None
    print("✔ Test 1 Passed: Vector store created")


# =========================================================
# TEST 2: RETRIEVAL WORKS
# =========================================================
def test_retrieval():
    query = "what is UML"
    docs = vectorstore.similarity_search(query, k=3)

    assert len(docs) > 0, "No documents retrieved"

    print("✔ Test 2 Passed: Retrieval working")
    print("Top result preview:")
    print(docs[0].page_content[:200])


# =========================================================
# TEST 3: METADATA PRESERVED (YOUR UPGRADE)
# =========================================================
def test_metadata():
    query = "UML diagram"
    docs = vectorstore.similarity_search(query, k=3)

    meta = docs[0].metadata

    assert isinstance(meta, dict), "Metadata missing"

    print("✔ Test 3 Passed: Metadata exists")
    print("Metadata sample:", meta)


# ----------------------------
# RUN TESTS
# ----------------------------
test_vectorstore_exists()
test_retrieval()
test_metadata()

print("\nRAG system ready. Type 'exit' to quit.\n")


# ----------------------------
# INTERACTIVE MODE
# ----------------------------
while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    try:
        answer = ask_question(vectorstore, query)
        print("\nAnswer:", answer)
    except Exception as e:
        print("\nError:", e)