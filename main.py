import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import our core RAG modules
from core.ingestion.document_loader import DocumentLoader
from core.ingestion.chunker import RecursiveCharacterChunker
from core.indexing.dense_index import DenseVectorStore

# 1. Initialize FastAPI app
app = FastAPI(
    title="SmartDocs - Advanced Multi-Strategy RAG Engine",
    description="Production-grade RAG API with hybrid search, reranking, and citation-backed Q&A.",
    version="1.0.0"
)

# 2. Setup upload directory and core instances
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
vector_store = DenseVectorStore(collection_name="smartdocs_main")

# 3. Pydantic Models for Type Safety
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    question: str
    results: List[SearchResultItem]

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    """Health check endpoint to verify the service is live."""
    return {
        "status": "online",
        "service": "SmartDocs RAG Engine",
        "version": "1.0.0"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a real file (PDF, TXT, DOCX, MD), extracts pages,
    chunks it with metadata, and indexes it into ChromaDB.
    """
    allowed_extensions = [".pdf", ".txt", ".md", ".docx"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    # 1. Save uploaded file to disk
    saved_file_path = UPLOAD_DIR / file.filename
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 2. Extract text and metadata
    try:
        loaded_docs = DocumentLoader.load_file(str(saved_file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")

    if not loaded_docs:
        raise HTTPException(status_code=400, detail="The uploaded document contains no readable text.")

    # 3. Chunk the document
    chunks = chunker.chunk_documents(loaded_docs)

    # 4. Index chunks into ChromaDB
    indexed_count = vector_store.add_chunks(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "pages_extracted": len(loaded_docs),
        "total_chunks_indexed": indexed_count,
        "message": f"Successfully ingested and indexed '{file.filename}' into SmartDocs!"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QuestionRequest):
    """
    Searches indexed documents for the closest matching chunks to the user's question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Search ChromaDB
    results = vector_store.search(query=request.question, top_k=request.top_k)

    return QueryResponse(
        question=request.question,
        results=results
    )