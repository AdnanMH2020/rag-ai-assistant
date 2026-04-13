import os

from app.rag.loader import load_and_split
from app.rag.vectorstore import load_or_create_vectorstore
from app.rag.retriever import retrieve
from app.rag.pipeline import ask_question


PDF_PATH = os.path.join("data", "SoftwareText.pdf")


def test_loader():
    print("\n[TEST] Loader")
    chunks = load_and_split(PDF_PATH)

    assert len(chunks) > 0, "No chunks created"
    assert hasattr(chunks[0], "page_content"), "Chunk missing page_content"

    print("✔ Loader works")


def test_vectorstore():
    print("\n[TEST] Vectorstore")
    chunks = load_and_split(PDF_PATH)

    vectorstore = load_or_create_vectorstore(chunks)

    results = vectorstore.similarity_search("UML diagrams", k=2)

    assert len(results) > 0, "Vector search failed"

    print("✔ Vectorstore works")


def test_retriever():
    print("\n[TEST] Retriever")
    chunks = load_and_split(PDF_PATH)
    vectorstore = load_or_create_vectorstore(chunks)

    docs = retrieve(vectorstore, "UML diagrams", k=2)

    assert len(docs) > 0, "Retriever returned nothing"

    print("✔ Retriever works")


def test_pipeline():
    print("\n[TEST] Pipeline")
    chunks = load_and_split(PDF_PATH)
    vectorstore = load_or_create_vectorstore(chunks)

    answer = ask_question(vectorstore, "What is UML?")

    assert answer is not None, "Pipeline returned None"
    assert len(answer) > 0, "Empty answer"

    print("✔ Pipeline works")


def run_all_tests():
    print("\n==============================")
    print("RUNNING RAG SYSTEM TESTS")
    print("==============================")

    test_loader()
    test_vectorstore()
    test_retriever()
    test_pipeline()

    print("\n==============================")
    print("ALL TESTS PASSED ✔")
    print("==============================")


if __name__ == "__main__":
    run_all_tests()