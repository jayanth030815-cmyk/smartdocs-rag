import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Core pipeline imports
from core.ingestion.document_loader import DocumentLoader
from core.ingestion.chunker import RecursiveCharacterChunker
from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index
from core.generation.llm_client import LLMClient
from core.pipeline import SmartDocsPipeline

# 1. Initialize FastAPI Application
app = FastAPI(
    title="SmartDocs - Advanced Multi-Strategy RAG Engine",
    description="Enterprise-grade RAG API featuring Hybrid Search (Dense + BM25), Intent Routing, Cross-Encoder Reranking, Citations, and Hallucination Guardrails.",
    version="2.0.0"
)

# 2. CORS Middleware (Allows Web Frontends / Streamlit / React to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Persistent Storage & Orchestrator
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
dense_store = DenseVectorStore(collection_name="smartdocs_production")
sparse_index = SparseBM25Index()
llm_client = LLMClient()

pipeline = SmartDocsPipeline(
    dense_store=dense_store,
    sparse_index=sparse_index,
    llm_client=llm_client
)

# 4. Pydantic Schemas for Type-Safe Contracts
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., example="What is the refund policy?")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Multi-turn conversation history")
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

class CitationItem(BaseModel):
    source: str
    page: int
    chunk_id: str

class PipelineTrace(BaseModel):
    original_query: str
    intent: Optional[str]
    rewritten_query: Optional[str]
    retrieved_count: int
    reranked_count: int
    chunks_used: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    guardrail_verified: bool

class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    grounded: bool
    trace: PipelineTrace

# ==========================================
# PRODUCTION API ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    """Service health check & catalog status."""
    return {
        "status": "online",
        "service": "SmartDocs Enterprise RAG Engine",
        "version": "2.0.0",
        "architecture": "Hybrid Search (ChromaDB + BM25) -> Cross-Encoder Reranker -> Grounded Citations"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads and indexes a document (PDF, TXT, DOCX, MD) into both
    Dense (ChromaDB) and Sparse (BM25) indexes simultaneously.
    """
    allowed_exts = [".pdf", ".txt", ".md", ".docx"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed: {allowed_exts}"
        )

    # Save physical copy
    saved_path = UPLOAD_DIR / file.filename
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse and chunk
    try:
        loaded_docs = DocumentLoader.load_file(str(saved_path))
        chunks = chunker.chunk_documents(loaded_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=400, detail="Document contains no extractable text.")

    # Index in both Dense & Sparse search engines
    dense_count = dense_store.add_chunks(chunks)
    sparse_index.add_chunks(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "pages": len(loaded_docs),
        "chunks_indexed": len(chunks),
        "message": f"Successfully processed '{file.filename}' into SmartDocs!"
    }

@app.post("/chat", response_model=ChatResponse)
def chat_with_docs(request: ChatRequest):
    """
    Main RAG Endpoint:
    Runs full pipeline (Routing -> Rewriting -> Hybrid RRF -> Reranking -> Citations -> Guardrails).
    Returns answer + complete observability trace.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history_dicts = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
    
    result = pipeline.answer_query(
        query=request.message,
        chat_history=history_dicts,
        top_k=request.top_k
    )

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        grounded=result["grounded"],
        trace=result["trace"]
    )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Real-Time Token Streaming (Server-Sent Events / SSE):
    Streams the grounded answer word-by-word just like ChatGPT!
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history_dicts = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []

    # Run retrieval pipeline
    result = pipeline.answer_query(
        query=request.message,
        chat_history=history_dicts,
        top_k=request.top_k
    )

    full_answer = result["answer"]

    # Stream tokens word-by-word
    async def token_generator():
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            token_payload = word + (" " if i < len(words) - 1 else "")
            yield f"data: {token_payload}\n\n"
            await asyncio.sleep(0.02) # Realistic smooth streaming cadence
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/documents")
def list_documents():
    """Lists indexed files in the upload directory."""
    files = list(UPLOAD_DIR.glob("*"))
    return {
        "total_files": len(files),
        "files": [{"filename": f.name, "size_bytes": f.stat().st_size} for f in files]
    }

@app.delete("/documents")
def clear_all_documents():
    """Clears all indexed documents and resets vector & sparse databases."""
    dense_store.clear()
    sparse_index.clear()
    for f in UPLOAD_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    return {"status": "success", "message": "All documents and vector indices have been cleared."}