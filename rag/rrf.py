from rag.retriever import retrieve
from rag.bm25 import get_all_chunks, document_frequency, inverse_document_frequency
from rag.bm25 import document_lengths, average_document_length, bm25_search



def rrf(vector_res, bm25_res, k=60):
    scores = {}

    for rank, res in enumerate(vector_res, start=1):
        chunk_id = res[0]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, res in enumerate(bm25_res, start=1):
        chunk_id = res[0]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    return scores