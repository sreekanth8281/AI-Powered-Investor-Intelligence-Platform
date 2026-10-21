from rag.retriever import retrieve
from rag.bm25 import BM25



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
    query = "What was Apple's net sales in 2024?"

    vector_res = retrieve(query, top_k=5)

    #BM25 search
    bm25 = BM25()
    bm25_res = bm25.search(query, top_k=5)

    #Combine the rankings
    scores = rrf(vector_res, bm25_res)

    print("\nVector search ranking")
    for rank, result in enumerate(vector_res, start=1):
        print(f"rank:{rank}, chunk_id: {result[0]}")

    print("\nBM25 ranking")
    for rank, result in enumerate(bm25_res, start=1):
        print(f"rank:{rank}, chunk_id: {result[0]}")


    print("\nRRF scores")

    for chunk_id, score in sorted(
        scores.items(),
        key=lambda x: x[1], reverse= True
    ):
        print(f"Chunk ID: {chunk_id}, Score: {score:.6f}")



     
    

