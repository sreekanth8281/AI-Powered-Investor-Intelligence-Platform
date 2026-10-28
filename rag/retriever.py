from ingestion.embeddings import get_embedding_model
from database.database import get_connection
from sklearn.metrics.pairwise import cosine_similarity


def diagnose_chunks():
    queries = [
        "What was Microsoft's revenue in fiscal year 2024?",
        "What was Microsoft's operating income in fiscal year 2024?",
        "What were Microsoft's major business segments in fiscal year 2024?"
    ]

    embedding_model = get_embedding_model()
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, chunk, embedding, company, year, source_file
                FROM document_chunks
                WHERE company ILIKE '%Microsoft%'
                ORDER BY id;
                """
            )

            rows = cursor.fetchall()

        print(f"Microsoft chunks found: {len(rows)}")

        if not rows:
            print("No Microsoft chunks found. Check company metadata.")
            return

        for query in queries:
            query_embedding = embedding_model.embed_query(query)

            print("\n" + "=" * 80)
            print(f"QUERY: {query}")
            print("=" * 80)

            scored_chunks = []

            for chunk_id, chunk_text, stored_embedding, company, year, source_file in rows:
                if stored_embedding is None:
                    continue

                stored_vector = stored_embedding.to_list()

                if len(stored_vector) != len(query_embedding):
                    continue

                similarity = cosine_similarity(
                    [query_embedding],
                    [stored_vector]
                )[0][0]

                scored_chunks.append(
                    (chunk_id, similarity, chunk_text, company, year, source_file)
                )

            scored_chunks.sort(key=lambda item: item[1], reverse=True)

            for rank, item in enumerate(scored_chunks[:5], start=1):
                chunk_id, similarity, chunk_text, company, year, source_file = item

                print(f"\nRank: {rank}")
                print(f"Chunk ID: {chunk_id}")
                print(f"Cosine similarity: {similarity:.6f}")
                print(f"Company: {company}, Year: {year}")
                print(f"Source: {source_file}")
                print(f"Chunk preview:\n{chunk_text[:500]}")

    finally:
        conn.close()


if __name__ == "__main__":
    diagnose_chunks()
