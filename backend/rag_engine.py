import sys, os
try:
    _BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BACKEND_DIR = os.getcwd()
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer   # replaces FlagEmbedding
from flashrank import Ranker, RerankRequest

# ── Singletons ────────────────────────────────────────────────
_chroma_client = None
_collection    = None
_embedder      = None
_ranker        = None

COLLECTION_NAME = "circuit_data"
EMBED_MODEL     = "BAAI/bge-m3"          # same model, different loader
RERANK_MODEL    = "ms-marco-MiniLM-L-6-v2"


def _get_chroma():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            _collection = _chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = _chroma_client.create_collection(COLLECTION_NAME)
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        print("[rag] Loading BGE-M3 via sentence-transformers...")
        # sentence-transformers loads BGE-M3 identically; fp16 on GPU is automatic
        _embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    return _embedder


def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=RERANK_MODEL, cache_dir="/tmp/flashrank")
    return _ranker


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────

def store_document(full_text: str, source: str = "uploaded_pdf") -> int:
    """Chunk full_text and upsert into ChromaDB. Returns number of chunks stored."""
    collection = _get_chroma()
    embedder   = _get_embedder()

    chunks = _chunk_text(full_text)
    # sentence-transformers returns numpy arrays — convert to list for ChromaDB
    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()

    ids       = [_stable_id(c) for c in chunks]
    metadatas = [{"source": source, "chunk_index": i} for i, c in enumerate(chunks)]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"[rag] Upserted {len(chunks)} chunks from '{source}'.")
    return len(chunks)


def retrieve_context(query: str, top_k: int = 10, rerank_top_n: int = 3) -> str:
    """
    Embed the query → fetch top_k candidates from ChromaDB →
    rerank with FlashRank → return top rerank_top_n joined as context.
    """
    collection = _get_chroma()
    embedder   = _get_embedder()
    ranker     = _get_ranker()

    count = collection.count()
    if count == 0:
        return ""

    query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "distances"],
    )

    candidates = results["documents"][0] if results["documents"] else []
    if not candidates:
        return ""

    rerank_request = RerankRequest(query=query, passages=[{"text": c} for c in candidates])
    ranked = ranker.rerank(rerank_request)

    top_passages = [r["text"] for r in ranked[:rerank_top_n]]
    return "\n\n---\n\n".join(top_passages)


def clear_collection():
    """Wipe all stored data (called before loading a new PDF)."""
    global _collection
    if _chroma_client and _collection:
        _chroma_client.delete_collection(COLLECTION_NAME)
        _collection = _chroma_client.create_collection(COLLECTION_NAME)
        print("[rag] Collection cleared.")
