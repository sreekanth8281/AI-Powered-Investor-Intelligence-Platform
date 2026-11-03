# AI-Powered Investor Intelligence Platform

A retrieval-augmented generation (RAG) backend for question answering over company financial reports. Annual reports are converted to Markdown, chunked, embedded, and stored in PostgreSQL with pgvector. At query time, the system runs semantic and lexical retrieval in parallel, fuses the rankings with Reciprocal Rank Fusion (RRF), and passes the top chunks to an LLM to produce a grounded answer.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Design Notes](#design-notes)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Author](#author)

## Overview

Financial reports mix narrative prose with dense numeric tables, so neither pure semantic search nor pure keyword search is sufficient on its own. This project combines both:

- **Semantic search** (pgvector) captures meaning when the query wording differs from the document.
- **BM25** captures exact terms such as line items ("operating income"), segment names, and fiscal years.
- **RRF** merges the two ranked lists without needing to calibrate their different score scales.

Retrieved chunks are then supplied to an LLM so that answers are based on the source document rather than on the model's prior knowledge.

## Key Features

- PDF-to-Markdown conversion of annual reports using PyMuPDF4LLM
- Semantic chunking with a token cap aligned to the embedding model's sequence limit
- 384-dimensional embeddings from `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage and cosine-distance search in PostgreSQL with pgvector
- BM25 lexical retrieval
- Reciprocal Rank Fusion of vector and BM25 results
- Company, fiscal year, and source-file metadata on every chunk
- Idempotent ingestion: previous rows for the same company, year, and source file are replaced on re-ingest, and exact duplicate chunks within a batch are skipped
- FastAPI service layer with interactive API documentation

## Architecture

### Query path

```mermaid
flowchart TD
    A[User query] --> B[FastAPI]
    B --> C[Hybrid retrieval]
    C --> D[Vector search<br/>pgvector, cosine distance]
    C --> E[BM25<br/>lexical search]
    D --> F[Reciprocal Rank Fusion]
    E --> F
    F --> G[Top-k context chunks]
    G --> H[LLM]
    H --> I[Grounded answer]
```

### Ingestion path

```mermaid
flowchart LR
    A[PDF] --> B[PyMuPDF4LLM]
    B --> C[Markdown]
    C --> D[Semantic chunking]
    D --> E[Sentence Transformer<br/>384-d embeddings]
    E --> F[(PostgreSQL + pgvector)]
```

Each stored row in `document_chunks` contains:

| Column        | Description                          |
|---------------|--------------------------------------|
| `id`          | Primary key                          |
| `chunk`       | Chunk text                           |
| `embedding`   | 384-dimensional pgvector embedding   |
| `company`     | Company name                         |
| `year`        | Fiscal year                          |
| `source_file` | Originating PDF                      |
| `metadata`    | Additional JSON metadata             |

## Tech Stack

| Area                | Technologies                                              |
|---------------------|-----------------------------------------------------------|
| Backend             | Python, FastAPI, Uvicorn                                  |
| Database            | PostgreSQL, pgvector, SQLAlchemy, psycopg2                |
| Retrieval / NLP     | Sentence Transformers, Hugging Face, BM25, RRF, LangChain |
| Document processing | PyMuPDF4LLM, Markdown                                     |
| Infrastructure      | Docker, environment-based configuration, Git              |

## Project Structure

```text
AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM/
├── app/                # FastAPI application
├── database/
│   └── database.py     # Connection, schema, insertion
├── ingestion/          # PDF -> Markdown -> chunks -> embeddings
├── rag/
│   ├── embeddings.py   # Embedding generation
│   ├── retriever.py    # Vector search and hybrid retrieval
│   └── bm25.py         # BM25 retrieval
├── data/
│   ├── pdf/            # Source reports
│   └── markdown/       # Converted reports
├── requirements.txt
├── Dockerfile
└── README.md
```

The layout may change as the project evolves.

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with the `pgvector` extension available
- An API key for the LLM provider used for answer generation

### Installation

```bash
git clone https://github.com/<your-username>/AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM.git
cd AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM

uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Database setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application stores chunks in the `document_chunks` table described above, using a `vector(384)` column for embeddings.

### Run the API

```bash
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

## Configuration

Create a `.env` file in the project root. Never commit it to version control.

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
LLM_API_KEY=your_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

## Usage

After ingesting a report, ask a question scoped by company and year:

```text
What was Microsoft's operating income in fiscal year 2024?
```

The system embeds the query, runs vector search and BM25 over the matching company and year, fuses the two rankings with RRF, and sends the top chunks plus the question to the LLM.

## Design Notes

**Why RRF.** Cosine distances and BM25 scores are on unrelated scales. RRF uses only rank positions, so the two lists can be combined without score normalization. A chunk found by both retrievers is favored over a chunk found by only one.

**Chunk size.** `all-MiniLM-L6-v2` truncates input beyond 256 tokens, so chunk sizes are capped below that limit to avoid silently losing content at embedding time.

**Cosine distance.** pgvector's `<=>` operator returns cosine distance (lower is closer). Similarity is `1 - distance`.

## Known Limitations

- **Tables in prose-oriented embeddings.** Numeric financial tables can embed poorly against natural-language questions, so exact figures may rank lower in vector search than narrative passages that mention the same terms. This is under active investigation, with retrieval diagnostics being run against Apple and Microsoft FY2024 reports.
- **Ingestion deduplication.** Duplicate chunk text is currently removed per ingestion batch, which may also collapse legitimately repeated text such as table headers.
- **No document upload endpoint yet.** Reports are ingested through the ingestion pipeline rather than a public upload API.
- **No evaluation harness yet.** Retrieval quality is not yet measured against a labeled question set.

## Roadmap

- [ ] Retrieval evaluation set and metrics (hit rate / MRR on financial questions)
- [ ] Improved table-aware chunking for financial statements
- [ ] Cross-encoder reranking
- [ ] Document upload API
- [ ] Citation-aware responses with source chunk references
- [ ] Financial metric extraction
- [ ] Company-to-company comparison
- [ ] Authentication and user management
- [ ] Conversation history and streaming responses
- [ ] Docker Compose setup
- [ ] Automated tests and CI/CD
- [ ] Cloud deployment, monitoring, and logging

## Author

**Sreekanth**
B.Tech, Computer Science Engineering

Built as a practical project exploring RAG systems, information retrieval, and production-style AI backends.