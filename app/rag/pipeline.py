from langchain_ollama import ChatOllama

def ask_question(vectorstore, query, k=6):

    docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return "No relevant context found in document."

    context = "\n\n".join([
        f"PAGE: {d.metadata.get('page')}\nCONTENT: {d.page_content}"
        for d in docs
    ])

    llm = ChatOllama(model="llama3")

    prompt = f"""
You are a precise AI assistant.

RULES:
- Use ONLY the provided context
- If not found, say "Not found in document"
- Do not hallucinate

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content