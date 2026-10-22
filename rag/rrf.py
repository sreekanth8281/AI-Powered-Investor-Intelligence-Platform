from rag.retriever import retrieve
from rag.bm25 import BM25
from database.vector_store import get_chunk_by_ids



def rrf(vector_res, bm25_res, k=60):
    scores = {}

    for rank, res in enumerate(vector_res, start=1):
        chunk_id = res[0]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, res in enumerate(bm25_res, start=1):
        chunk_id = res[0]
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    return scores

if __name__ == '__main__':
    query = "What were Apple's operating expenses in 2024?"

    vector_res = retrieve(query, top_k=5)

    #BM25 search
    bm25 = BM25()
    bm25_res = bm25.search(query, top_k=5)

    #Combine the rankings
    scores = rrf(vector_res, bm25_res)
    print("\n--- VECTOR RESULTS ---")

    for rank, row in enumerate(vector_res, start=1):
        print(f"Rank: {rank}, Chunk ID: {row[0]}")

    print("\n--- BM25 RESULTS ---")

    for rank, row in enumerate(bm25_res, start=1):
        print(f"Rank: {rank}, Chunk ID: {row[0]}")




    res =  sorted(
        scores.items(),
        key=lambda x: x[1], reverse= True
    )

    top_res = res[:5]

    chunk_ids = [
        chunk_id for chunk_id, score in top_res
    ]

    chunks = get_chunk_by_ids(chunk_ids)

    chunks_by_id = {
        row[0]: row 
        for row in chunks
    }

    for chunk_id, score in top_res:
        row = chunks_by_id.get(chunk_id)

        if row is None:
            continue

    #print(query)
    #print("*"*80)
    #print("Chunk ID:", row[0])
    #print("RRF Score:", score)
    #print("Company:", row[2])
    #print("Year:", row[3])
    #print("Source:", row[4])
    #print("Chunk Text:")
    #print(row[1])
    #print("*" * 80)



     
    

