def retrieve(vectorstore, query, k=4):
    """
    Returns top-k relevant documents from the vector store.
    """

    docs = vectorstore.similarity_search(query, k=k)

    return docs