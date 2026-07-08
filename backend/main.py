import os
import shutil
import traceback
from fastapi import Query
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.document_ingestion import (
    partition_document,
    chunk_document,
    process_chunks,
    create_vectorstore,
)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
PERSIST_DIR = "./chroma_db"
SCORE_THRESHOLD = 1.0
TOP_K = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatOllama(model="llama3.2:latest")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG Backend Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        print("1 Upload started")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Sanitize filename to avoid path traversal via "../" etc.
        safe_filename = os.path.basename(file.filename)

        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)

        print("2 Saving file")

        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("3 Partitioning")
        elements = partition_document(file_path)

        print("4 Chunking")
        chunks = chunk_document(elements)

        print("5 Processing")
        processed_documents = process_chunks(chunks, source_name=safe_filename)

        if not processed_documents:
            raise HTTPException(
                status_code=422,
                detail="No usable content could be extracted from this PDF",
            )

        print("6 Creating vectorstore")
        create_vectorstore(processed_documents)

        print("7 Done")

        return {
            "message": "PDF processed successfully",
            "chunks_indexed": len(processed_documents),
        }

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process PDF")


@app.post("/chat")
async def chat(question: str = Query(..., description="The question to ask")):
    question = question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    if not os.path.exists(PERSIST_DIR):
        raise HTTPException(
            status_code=400,
            detail="No document has been ingested yet. Upload a PDF first.",
        )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Retrieve top-k chunks
    results_with_scores = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    # Debug: print retrieved chunks
    print("=" * 80)
    print("Retrieved Chunks")
    print("=" * 80)

    for i, (doc, score) in enumerate(results_with_scores, start=1):
        print(f"\nChunk {i}")
        print(f"Distance: {score:.4f}")
        print("-" * 40)
        print(doc.page_content[:300])
        print("-" * 40)

    if not results_with_scores:
        return {
            "answer": "Information not found in notes.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(results_with_scores, start=1):

        source = doc.metadata.get("source", "unknown")
        pages = doc.metadata.get("pages", "unknown")

        context_parts.append(
            f"[Source {i} - {source}, page(s) {pages}]\n{doc.page_content}"
        )

        sources.append(
            {
                "source": source,
                "pages": pages,
                "distance": round(float(score), 4),
            }
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an academic assistant.

Answer ONLY using the information provided below.

Rules:

1. Use ONLY the provided information.
2. Do NOT use outside knowledge.
3. If the answer exists, answer it clearly.
4. Cite sources inline like [Source 1].
5. If the answer is not present, reply exactly:

Information not found in notes.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    print("=" * 80)
    print("Prompt Sent To LLM")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }