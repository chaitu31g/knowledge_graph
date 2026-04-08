"""
CircuitAI FastAPI Backend
Run via the Colab notebook cells — do not run this directly.
"""

# ── Path bootstrap (MUST be first) ────────────────────────────
# Ensures pdf_parser, rag_engine, model_loader are always found
# regardless of which directory Python was launched from.
import sys, os
try:
    _BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Running inside a Jupyter/Colab cell — __file__ is not defined.
    # Cell 0 sets os.chdir() to the backend dir, so getcwd() gives the right path.
    _BACKEND_DIR = os.getcwd()
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
# ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_parser import extract_all_pages, generate_chat_response, extract_specs
from rag_engine import store_document, retrieve_context, clear_collection

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="CircuitAI Backend",
    description="Vision-RAG API for semiconductor datasheet analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Cloudflare tunnel + local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──────────────────────────────────────────────────────

class AppState:
    latest_full_text: str = ""
    latest_specs: dict = {"Vgs": "N/A", "Id": "N/A", "Rdson": "N/A"}

state = AppState()

# ── Request / Response Models ──────────────────────────────────

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    specs: dict

class ProcessResponse(BaseModel):
    status: str
    message: str
    chunks_stored: int

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — the frontend pings this to verify connectivity."""
    return {"status": "ok", "model": "Qwen2.5-VL-7B-Instruct (4-bit)"}


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF, parse it page-by-page with the vision model,
    store chunks in ChromaDB, and extract sidebar specs.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    # Wipe previous document before loading a new one
    clear_collection()

    print(f"[/process] Received '{file.filename}' ({len(pdf_bytes) / 1024:.1f} KB)")

    # 1. Vision extraction
    full_text = extract_all_pages(pdf_bytes)
    if not full_text.strip():
        raise HTTPException(status_code=422, detail="No extractable content found in PDF.")

    state.latest_full_text = full_text

    # 2. Store in ChromaDB
    n_chunks = store_document(full_text, source=file.filename)

    # 3. Extract sidebar specs
    state.latest_specs = extract_specs(full_text)

    return ProcessResponse(
        status="success",
        message=f"Parsed and indexed '{file.filename}'.",
        chunks_stored=n_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    RAG-powered Q&A: retrieve relevant context → generate answer.
    """
    if not state.latest_full_text:
        raise HTTPException(
            status_code=400,
            detail="No datasheet has been processed yet. Upload a PDF first."
        )

    # Retrieve top-3 reranked chunks from ChromaDB
    context = retrieve_context(req.query)

    # Fallback to full text summary if retrieval returns nothing
    if not context:
        context = state.latest_full_text[:8000]

    response_text = generate_chat_response(context, req.query)

    return ChatResponse(response=response_text, specs=state.latest_specs)


# ── Server Entry (direct run only) ────────────────────────────
# When running on Colab, the notebook starts uvicorn directly.
# This block is only used for local testing outside Colab.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
