import os
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.bm25 import BM25
from rag.hybrid import hybrid_search
from rag.scope import detect_scope
from database.vector_store import get_available_documents
from ingestion.embeddings import get_embedding_model

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using ONLY "
    "the provided context from company annual reports. Quote figures "
    "exactly as written and include units. Cite the sources you used as "
    "[1], [2], etc. If the context does not contain the answer, say "
    "'I could not find this in the provided documents.' Do not guess."
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the embedding model and build the BM25 index.
    get_embedding_model()
    state["documents"] = get_available_documents()  # [(company, year), ...]
    state["bm25_cache"] = {}
    yield


app = FastAPI(title="Investor Intelligence Platform", lifespan=lifespan)


def get_bm25(company, year):
    """One BM25 index per (company, year) scope, built on first use."""
    key = (company.lower() if company else None, year)
    cache = state["bm25_cache"]
    if key not in cache:
        cache[key] = BM25(company=company, year=year)
    return cache[key]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=10)
    company: str | None = Field(None, description="e.g. Microsoft. Auto-detected from the question if omitted.")
    year: int | None = Field(None, description="e.g. 2024. Auto-detected from the question if omitted.")


class Source(BaseModel):
    chunk_id: int
    company: str
    year: int
    source_file: str
    rrf_score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    company_filter: str | None = None
    year_filter: int | None = None
    sources: list[Source]


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c['company']} {c['year']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    detected_company, detected_year = detect_scope(
        req.question, state["documents"]
    )
    company = req.company or detected_company   # explicit value wins
    year = req.year or detected_year

    chunks = hybrid_search(
        req.question,
        get_bm25(company, year),
        top_k=req.top_k,
        company=company,
        year=year,
    )

    if not chunks:
        return AskResponse(
            answer="I could not find this in the provided documents.",
            company_filter=company,
            year_filter=year,
            sources=[],
        )

    prompt = (
        f"Context:\n{build_context(chunks)}\n\n"
        f"Question: {req.question}"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096},
            },
            timeout=180,  # the first call is slow while the model loads
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Ollama. Is it running? Try: ollama serve",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return AskResponse(
        answer=answer,
        company_filter=company,
        year_filter=year,
        sources=chunks,
    )