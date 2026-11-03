import numpy as np

from ingestion.embeddings import get_embedding_model
from database.database import get_connection


QUERIES = [
    "What was Microsoft's revenue in fiscal year 2024?",
    "What was Microsoft's operating income in fiscal year 2024?",
    "What were Microsoft's major business segments in fiscal year 2024?",
]


def main():
    model = get_embedding_model()

    conn = get_connection()

    try:
        for query in QUERIES:
            print("\n" + "=" * 80)
            print("QUERY:", query)
            print("=" * 80)

            query_embedding = np.asarray(
                model.embed_query(query),
                dtype=np.float32,
            )

            # Retrieve the top 10 Microsoft chunks using PostgreSQL.
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        chunk,
                        embedding <=> %s::vector AS distance
                    FROM document_chunks
                    WHERE company ILIKE %s
                      AND year = %s
                      AND source_file = %s
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10
                    """,
                    (
                        query_embedding.tolist(),
                        "%Microsoft%",
                        2024,
                        "microsoft_report_2024_fixed.pdf",
                        query_embedding.tolist(),
                    ),
                )

                results = cursor.fetchall()

            for rank, (chunk_id, chunk, distance) in enumerate(
                results, start=1
            ):
                similarity = 1.0 - float(distance)

                print(f"\nRank: {rank}")
                print(f"Chunk ID: {chunk_id}")
                print(f"Cosine distance: {float(distance):.6f}")
                print(f"Cosine similarity: {similarity:.6f}")
                print(f"Chunk preview: {chunk[:250]!r}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
