from ingestion.embeddings import get_embedding_model
from database.vector_store import search, get_chunk_by_ids
from database.database import get_connection

from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query: str, top_k: int = 10):
    embedding_model = get_embedding_model()

    query_embedding = embedding_model.embed_query(query)

    print(f"Embedding dimensions: {len(query_embedding)}")

    results = search(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return results


if __name__ == "__main__":

    queries = [
        "What were Apple's operating expenses in 2024?",
        "Total operating expenses",
        "Research and development selling general and administrative total operating expenses"
    ]

    for query in queries:
        results = retrieve(query, top_k=10)

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        for rank, result in enumerate(results, start=1):
            print(
                f"Rank: {rank} | "
                f"ID: {result[0]} | "
                f"Distance: {result[-1]:.4f}"
            )

        target = next(
            (result for result in results if result[0] == 307),
            None
        )

        if target:
            rank = next(
                i for i, result in enumerate(results, start=1)
                if result[0] == 307
            )
            print(f"\nChunk 307 rank: {rank}")
            print(f"Chunk 307 distance: {target[-1]:.4f}")
        else:
            print("\nChunk 307 is not in the top 10.")
    

