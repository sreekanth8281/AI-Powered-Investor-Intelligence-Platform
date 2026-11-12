import os 
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.bm25 import BM25
from rag.hybrid import hybrid_search
from database.vector_store import 
from ingestion.embeddings import get_embedding_model
