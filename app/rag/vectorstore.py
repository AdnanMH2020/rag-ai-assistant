import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "vectorstore/faiss_index"


# ---------- EMBEDDINGS ----------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------- CREATE NEW STORE ----------
def create_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


# ---------- LOAD OR CREATE ----------
def load_or_create_vectorstore(chunks):
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        print("Loading existing vector store...")
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Creating new vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore


# ---------- ADD NEW DOCUMENTS ----------
def add_documents(vectorstore, new_chunks):
    embeddings = get_embeddings()

    new_store = FAISS.from_documents(new_chunks, embeddings)
    vectorstore.merge_from(new_store)

    vectorstore.save_local(INDEX_PATH)

    return vectorstore