import time
import os

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.rag.loader import load_and_split
from app.rag.vectorstore import load_or_create_vectorstore
from app.services.rag_service import answer_question

global vectorstore, chunks

# -------------------------
# App setup
# -------------------------
app = FastAPI(title="GenAI RAG System")

# Setup the template engine to look for HTML files in the 'templates' folder
templates = Jinja2Templates(directory="templates")


# -------------------------
# Request schema
# -------------------------
class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore

    # Save file to folder
    file_path = os.path.join("data", file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    #Process the uploaded file
    chunks = load_and_split(file_path)
    vectorstore = load_or_create_vectorstore(chunks)

    return {"message": f"{file.filename} uploaded and processed successfully"}


# -------------------------
# User Interface Route
# -------------------------
@app.get("/", response_class=HTMLResponse)
def render_home(request: Request):
    # Pass the context dictionary explicitly
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}
    )


# -------------------------
# Main RAG API endpoint
# -------------------------
@app.post("/ask")
def ask(req: QueryRequest):
    """
    Handles the background JSON communication.
    The HTML page calls this when you click 'Ask'.
    """
    if vectorstore is None:
        return {"answer": "Error: Vectorstore not initialized. Check PDF path.", "latency_seconds": 0}

    start_time = time.time()

    # Core RAG call (retrieval + LLM)
    answer = answer_question(vectorstore, req.question)

    end_time = time.time()
    latency = end_time - start_time

    print(f"[LATENCY] /ask took {latency:.4f}s")

    return {
        "question": req.question,
        "answer": answer,
        "latency_seconds": round(latency, 4)
    }