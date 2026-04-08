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

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_parser import extract_all_pages, generate_chat_response, extract_specs
from rag_engine import store_document, retrieve_context, clear_collection, _get_embedder
from model_loader import load_model

# ── Lifespan: pre-load models at startup ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on server startup — loads all heavy models before serving requests."""
    print("[startup] Pre-loading Qwen2.5-VL-3B into GPU...")
    load_model()                  # loads vision LLM + processor
    print("[startup] Pre-loading BGE-M3 embedding model...")
    _get_embedder()               # loads sentence-transformers model
    print("[startup] ✅ All models ready. Server is now accepting requests.")
    yield                         # server runs here
    print("[shutdown] Server shutting down.")

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="CircuitAI Backend",
    description="Vision-RAG API for semiconductor datasheet analysis",
    version="1.0.0",
    lifespan=lifespan,            # ← attach the startup/shutdown hook
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
    return {"status": "ok", "model": "Qwen2.5-VL-3B-Instruct (4-bit)"}


import asyncio, json, torch
from fastapi.responses import StreamingResponse
from pdf_parser import pdf_to_images, extract_text_from_page, extract_specs, generate_chat_response

def _sse(data: dict) -> str:
    """Format a dict as a single SSE message."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Stream real-time progress events (SSE) while parsing the PDF.
    Each yielded line is a JSON object with { type, message, ... }.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    filename  = file.filename
    loop      = asyncio.get_event_loop()

    async def progress_stream():
        try:
            # ── Step 0: clear previous doc ────────────────────
            await loop.run_in_executor(None, clear_collection)

            # ── Step 1: PDF → images ───────────────────────────
            yield _sse({"type": "progress", "step": "convert",
                        "message": f"Converting '{filename}' to page images..."})
            images = await loop.run_in_executor(None, pdf_to_images, pdf_bytes)
            yield _sse({"type": "progress", "step": "convert",
                        "message": f"Found {len(images)} pages. Starting vision extraction..."})

            # ── Step 2: vision extraction per page ────────────
            pages_text = []
            for i, img in enumerate(images):
                yield _sse({"type": "progress", "step": "extract",
                            "message": f"Extracting data from page {i+1} of {len(images)}...",
                            "page": i + 1, "total": len(images)})
                text = await loop.run_in_executor(None, extract_text_from_page, img)
                if text:
                    pages_text.append(f"### Page {i + 1}\n\n{text}")
                await loop.run_in_executor(None, torch.cuda.empty_cache)

            full_text = "\n\n---\n\n".join(pages_text)
            if not full_text.strip():
                yield _sse({"type": "error", "message": "No extractable content found in PDF."})
                return

            state.latest_full_text = full_text

            # ── Step 3: chunk + index ──────────────────────────
            yield _sse({"type": "progress", "step": "chunk",
                        "message": "Chunking text and building vector index..."})
            n_chunks = await loop.run_in_executor(None, store_document, full_text, filename)
            yield _sse({"type": "progress", "step": "chunk",
                        "message": f"Indexed {n_chunks} chunks into ChromaDB."})

            # ── Step 4: spec extraction ────────────────────────
            yield _sse({"type": "progress", "step": "specs",
                        "message": "Extracting component specifications for sidebar..."})
            specs = await loop.run_in_executor(None, extract_specs, full_text)
            state.latest_specs = specs

            # ── Done ───────────────────────────────────────────
            yield _sse({"type": "done",
                        "message": f"✅ Processed {len(images)} pages · {n_chunks} chunks indexed.",
                        "specs": specs})

        except Exception as e:
            yield _sse({"type": "error", "message": f"Processing failed: {str(e)}"})

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
