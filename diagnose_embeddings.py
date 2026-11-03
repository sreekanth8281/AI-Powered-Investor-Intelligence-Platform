import hashlib
import logging

import numpy as np

from ingestion.embeddings import get_embedding_model
from database.database import get_connection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_microsoft_chunks():
    """
    Fetch Microsoft 2024 chunks and their stored embeddings.
    This function does not modify the database.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    chunk,
                    embedding,
                    company,
                    year,
                    source_file
                FROM document_chunks
                WHERE company ILIKE %s
                  AND year = %s
                  AND source_file = %s
                  AND embedding IS NOT NULL
                ORDER BY id
                """,
                (
                    "%Microsoft%",
                    2024,
                    "microsoft_report_2024_fixed.pdf",
                ),
            )

            return cursor.fetchall()

    finally:
        conn.close()


def to_numpy_vector(vector):
    """
    Convert a pgvector Vector object into a NumPy array.
    """

    return np.asarray(vector.to_list(), dtype=np.float32)


def cosine_similarity_rows(a, b):
    """
    Calculate cosine similarity between corresponding rows
    of two matrices.
    """

    numerator = np.sum(a * b, axis=1)

    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )

    return numerator / np.maximum(denominator, 1e-12)


def diagnose_embeddings():
    print("=" * 75)
    print("EMBEDDING DIAGNOSTICS")
    print("=" * 75)

    # 1. Load the embedding model.
    model = get_embedding_model()

    print("\n[1] Embedding model loaded.")

    # 2. Fetch rows from PostgreSQL.
    rows = fetch_microsoft_chunks()

    if not rows:
        print("No matching Microsoft chunks found.")
        return

    chunk_ids = [row[0] for row in rows]
    chunks = [row[1] for row in rows]
    stored_vectors = [row[2] for row in rows]

    print(f"\n[2] Rows fetched: {len(rows)}")

    # 3. Convert stored pgvector objects into NumPy arrays.
    stored = np.stack(
        [to_numpy_vector(vector) for vector in stored_vectors]
    )

    print(f"Stored embedding shape: {stored.shape}")

    # 4. Count unique chunk texts.
    unique_hashes = {
        hashlib.md5(chunk.encode("utf-8")).hexdigest()
        for chunk in chunks
    }

    print(f"Distinct chunk texts: {len(unique_hashes)}")
    print(f"Repeated-text rows: {len(chunks) - len(unique_hashes)}")

    # 5. Generate fresh embeddings for the exact stored text.
    print("\n[3] Generating fresh embeddings...")

    fresh = np.asarray(
        model.embed_documents(chunks),
        dtype=np.float32,
    )

    print(f"Fresh embedding shape: {fresh.shape}")

    # 6. Verify dimensions.
    if stored.shape != fresh.shape:
        print("\nERROR: Stored and fresh embedding shapes differ.")
        print("Stored:", stored.shape)
        print("Fresh:", fresh.shape)
        return

    # 7. Compare each stored vector with its fresh counterpart.
    similarities = cosine_similarity_rows(stored, fresh)

    print("\n[4] STORED VS FRESH EMBEDDINGS")
    print("-" * 50)
    print(f"Minimum cosine similarity: {similarities.min():.8f}")
    print(f"Mean cosine similarity:    {similarities.mean():.8f}")
    print(f"Maximum cosine similarity: {similarities.max():.8f}")

    print("\nTen lowest-matching chunks:")
    print("-" * 75)

    for i in np.argsort(similarities)[:10]:
        print(f"\nChunk ID: {chunk_ids[i]}")
        print(f"Cosine similarity: {similarities[i]:.8f}")
        print(f"Chunk preview: {chunks[i][:200]!r}")

    # 8. Compare query embeddings against stored and fresh vectors.
    queries = [
        "What was Microsoft's revenue in fiscal year 2024?",
        "What was Microsoft's operating income in fiscal year 2024?",
        "What were Microsoft's major business segments in fiscal year 2024?",
    ]

    print("\n[5] QUERY RETRIEVAL COMPARISON")
    print("=" * 75)

    for query in queries:
        print(f"\nQUERY: {query}")

        query_embedding = np.asarray(
            model.embed_query(query),
            dtype=np.float32,
        )

        # Cosine similarity against the stored vectors.
        stored_scores = cosine_similarity_rows(
            stored,
            np.broadcast_to(query_embedding, stored.shape),
        )

        # Cosine similarity against the freshly generated vectors.
        fresh_scores = cosine_similarity_rows(
            fresh,
            np.broadcast_to(query_embedding, fresh.shape),
        )

        stored_top = np.argsort(stored_scores)[::-1][:5]
        fresh_top = np.argsort(fresh_scores)[::-1][:5]

        print("\nTop 5 using STORED embeddings:")

        for rank, i in enumerate(stored_top, start=1):
            print(
                f"Rank {rank} | "
                f"ID: {chunk_ids[i]} | "
                f"Similarity: {stored_scores[i]:.6f} | "
                f"Chunk: {chunks[i][:150]!r}"
            )

        print("\nTop 5 using FRESH embeddings:")

        for rank, i in enumerate(fresh_top, start=1):
            print(
                f"Rank {rank} | "
                f"ID: {chunk_ids[i]} | "
                f"Similarity: {fresh_scores[i]:.6f} | "
                f"Chunk: {chunks[i][:150]!r}"
            )

    print("\n" + "=" * 75)
    print("DIAGNOSTICS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    diagnose_embeddings()
