from ingestion.embeddings import get_embedding_model
from database.vector_store import search, get_chunk_by_ids
from database.database import get_connection


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
    query = "What were Apple's operating expenses in 2024?"

    # Generate the query embedding
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)

    # Run vector retrieval
    results = search(
        query_embedding=query_embedding,
        top_k=10
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunks\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print("ID:", result[0])
        print("Chunk:", result[1][:500])
        print("Company:", result[2])
        print("Year:", result[3])
        print("Distance:", result[-1])
        print("-" * 80)

    # Check whether the known relevant chunk exists
    chunk_307 = get_chunk_by_ids([307])
    print("\nCorrect chunk exists:", bool(chunk_307))

    # Calculate the distance between the query and chunk 307
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE id = %s;
                """,
                (query_embedding, 307)
            )

            chunk_distance = cursor.fetchone()

            print("\nRelevant chunk distance:")
            print(chunk_distance)

    finally:
        conn.close()

