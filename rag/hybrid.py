from rag.retriever import retrieve
from rag.rrf import rrf
from database.vector_store import get_chunk_by_ids


def hybrid_search(query, bm25, top_k = 5, candidates=20):
    vector_res = retrieve(query, top_k= candidates)
    bm25_res = bm25.search(query, top_k=candidates)

    scores = rrf(vector_res, bm25_res)
    ranked = sorted(scores.items(), key = lambda x: x[1], reverse=True)[:top_k]
    rows = {r[0]: r for r in get_chunk_by_ids([cid for cid, _ in ranked])}

    results = []
    for chunk_id, score in ranked:
        row = rows.get(chunk_id)
        if row is None:
            continue
        results.append({
            "chunk_id": row[0],
            "text": row[1],
            "company": row[2],
            "year": row[3],
            "source_file": row[4],
            "rrf_score": round(score, 6),
        })

    return results
