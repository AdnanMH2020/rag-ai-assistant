from app.rag.loader import load_and_split
from app.rag.vectorstore import create_vector_store


def test_vectorstore_creation():
    chunks = load_and_split("data/SoftwareText.pdf")
    vectorstore = create_vector_store(chunks)

    assert vectorstore is not None


def test_retrieval():
    chunks = load_and_split("data/SoftwareText.pdf")
    vectorstore = create_vector_store(chunks)

    docs = vectorstore.similarity_search("UML", k=3)

    assert len(docs) > 0

if __name__ == "__main__":
    test_vectorstore_creation()
    test_retrieval()