def answer_question(vectorstore, query):
    """
    Service layer:
    - connects API → RAG pipeline
    - keeps API clean
    """

    from app.rag.pipeline import ask_question

    return ask_question(vectorstore, query)