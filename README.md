# 📊 AI-Powered Investor Intelligence Platform

An AI-powered financial document intelligence platform that allows users to upload company reports and ask natural-language questions about them.

The system uses **Retrieval-Augmented Generation (RAG)** with **semantic vector search + BM25 lexical search + Reciprocal Rank Fusion (RRF)** to retrieve relevant information before generating grounded answers with an LLM.

---

## 🚀 Features

* 📄 Upload and process financial documents such as annual reports
* 🔄 Convert PDF documents into structured Markdown
* ✂️ Semantic document chunking
* 🧠 Generate vector embeddings using Hugging Face Sentence Transformers
* 🔎 **Hybrid Retrieval**

  * Semantic/vector search using PostgreSQL + pgvector
  * Lexical retrieval using BM25
  * Reciprocal Rank Fusion (RRF) to combine rankings
* 🤖 LLM-powered question answering
* 📚 Context-aware responses based on retrieved document chunks
* 🏢 Company and year metadata filtering
* 🐘 PostgreSQL database with `pgvector`
* ⚡ FastAPI backend
* 🐳 Docker-ready architecture

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │      User Query     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   FastAPI Backend   │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │     Hybrid Retrieval      │
              └─────────────┬─────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ Vector Search   │       │   BM25 Search   │
      │   pgvector      │       │ Lexical Search  │
      └────────┬────────┘       └────────┬────────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                 ┌─────────────────────┐
                 │        RRF          │
                 │ Rank Fusion         │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Retrieved Context   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       LLM           │
                 │   Answer Generation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Final Response    │
                 └─────────────────────┘
```

---

## 📥 Document Ingestion Pipeline

When a financial report is uploaded, it goes through the following pipeline:

```text
PDF
 │
 ▼
PyMuPDF4LLM
 │
 ▼
Markdown
 │
 ▼
Semantic Chunking
 │
 ▼
Sentence Transformer
 │
 ▼
Embeddings
 │
 ▼
PostgreSQL + pgvector
```

Each chunk is stored along with metadata such as:

* Company
* Year
* Source file
* Chunk content
* Embedding vector
* Additional metadata

---

## 🔎 Hybrid Retrieval

The platform combines two different retrieval techniques.

### 1. Semantic Search

The user's query is converted into an embedding using a Sentence Transformer.

The system then searches PostgreSQL using `pgvector` to find semantically similar document chunks.

```text
Query
  ↓
Embedding
  ↓
pgvector similarity search
  ↓
Top-K semantic results
```

This helps retrieve information even when the query uses different wording from the document.

### 2. BM25 Lexical Search

BM25 retrieves documents based on the actual words appearing in the query and documents.

It is useful for exact terms such as:

```text
"iPhone"
"revenue"
"operating income"
"Tesla Model 3"
"2024"
```

### 3. Reciprocal Rank Fusion

The semantic and BM25 results are combined using **Reciprocal Rank Fusion (RRF)**.

Conceptually:

```text
Vector Results ─────┐
                    ├──► RRF ──► Final Ranked Results
BM25 Results ───────┘
```

This allows the system to benefit from both:

* Semantic similarity
* Exact keyword matching

---

## 🤖 Retrieval-Augmented Generation

Instead of sending the entire document to the LLM, the system retrieves only the most relevant chunks.

```text
User Question
      │
      ▼
Hybrid Retrieval
      │
      ▼
Relevant Chunks
      │
      ▼
Context + Question
      │
      ▼
LLM
      │
      ▼
Grounded Answer
```

This reduces unnecessary context and helps the model answer questions using information retrieved from the uploaded documents.

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Database

* PostgreSQL
* pgvector
* SQLAlchemy
* psycopg2

### AI / NLP

* Sentence Transformers
* Hugging Face
* LangChain
* Semantic Chunking
* BM25
* RAG
* LLM API

### Document Processing

* PyMuPDF4LLM
* Markdown

### Infrastructure

* Docker
* Environment variables
* Git & GitHub

---

## 📂 Project Structure

```text
AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM/
│
├── data/
│   ├── markdown/
│   └── pdf/
│
├── database/
│   └── database.py
│
├── rag/
│   ├── embeddings.py
│   ├── retriever.py
│   ├── bm25.py
│   └── ...
│
├── ingestion/
│   └── ...
│
├── app/
│   └── ...
│
├── requirements.txt
├── .env
├── .gitignore
├── Dockerfile
└── README.md
```

> The exact folder structure may change as the project evolves.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM.git

cd AI-POWERED-INVESTOR-INTELLIGENCE-PLATFORM
```

### 2. Create a virtual environment

Using `uv`:

```bash
uv venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

LLM_API_KEY=your_api_key

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

EMBEDDING_DEVICE=cpu
```

Never commit your `.env` file to GitHub.

---

## 🐘 PostgreSQL Setup

Install PostgreSQL and enable the `pgvector` extension.

Example:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application stores document chunks and their embeddings in PostgreSQL.

Example structure:

```text
document_chunks
├── id
├── chunk
├── embedding
├── company
├── year
├── source_file
└── metadata
```

The current embedding model produces **384-dimensional embeddings**.

---

## ▶️ Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Query

After ingesting financial reports, users can ask questions such as:

```text
What was Apple's revenue in 2024?
```

The system performs:

```text
Question
   ↓
Query preprocessing
   ↓
Embedding generation
   ↓
Vector search
   +
BM25 search
   ↓
RRF ranking
   ↓
Top relevant chunks
   ↓
LLM
   ↓
Answer
```

Example:

```text
User:
What was Apple's revenue in 2024?

System:
1. Searches the vector database
2. Performs BM25 retrieval
3. Combines results using RRF
4. Retrieves relevant Apple 2024 report chunks
5. Sends the context to the LLM
6. Generates the final answer
```

---

## 🎯 Why Hybrid Search?

Pure vector search can miss exact terminology.

Pure keyword search can miss semantically related information.

Hybrid retrieval combines both approaches:

```text
                 Hybrid Search
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Semantic Search           BM25 Search
          │                       │
          └───────────┬───────────┘
                      ▼
                     RRF
                      │
                      ▼
              Better Retrieval
```

This is particularly useful for financial documents where both **meaning** and **exact terminology** matter.

---

## 📈 Current Capabilities

* PDF → Markdown conversion
* Semantic document chunking
* Embedding generation
* PostgreSQL vector storage
* Vector similarity search
* BM25 lexical retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion
* RAG-based question answering

---

## 🔮 Future Improvements

* [ ] Add authentication and user management
* [ ] Add document upload API
* [ ] Add company comparison
* [ ] Add financial metric extraction
* [ ] Add citation-aware responses
* [ ] Add reranking using a cross-encoder
* [ ] Add conversation history
* [ ] Add streaming LLM responses
* [ ] Add Docker Compose
* [ ] Add automated tests
* [ ] Add CI/CD pipeline
* [ ] Deploy the application to the cloud
* [ ] Add monitoring and logging

---

## 🧠 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Vector databases
* Embeddings
* Semantic search
* Lexical retrieval
* BM25
* Reciprocal Rank Fusion
* PostgreSQL + pgvector
* FastAPI
* Document processing
* LLM integration
* Hybrid information retrieval

---

## 👨‍💻 Author

**Sreekanth**

B.Tech Computer Science Engineering

Built as a practical project to explore **RAG systems, information ret**
