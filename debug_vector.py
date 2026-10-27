from ingestion.embeddings import get_embedding_model
from database.vector_store import get_connection

embedding_model = get_embedding_model()
conn = get_connection()

try:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, chunk, embedding
            FROM document_chunks
            WHERE id = %s;
            """,
            (307,)
        )
        row = cursor.fetchone()

    if row is None:
        print("ERROR: Chunk 307 was not found.")

    else:
        chunk_id, chunk_text, stored_embedding = row

        # Convert PostgreSQL Vector to a Python list
        stored_embedding = stored_embedding.to_list()

        # Generate a fresh embedding for the same text
        fresh_embedding = embedding_model.embed_documents(
            [chunk_text]
        )[0]

        print("Chunk ID:", chunk_id)
        print("Text preview:", chunk_text[:300])
        print("Stored dimensions:", len(stored_embedding))
        print("Fresh dimensions:", len(fresh_embedding))

        # Calculate cosine similarity
        dot_product = sum(
            a * b
            for a, b in zip(stored_embedding, fresh_embedding)
        )

        stored_norm = sum(
            a * a for a in stored_embedding
        ) ** 0.5

        fresh_norm = sum(
            a * a for a in fresh_embedding
        ) ** 0.5

        cosine_similarity = dot_product / (
            stored_norm * fresh_norm
        )

        print(
            "Stored vs fresh cosine similarity:",
            round(cosine_similarity, 6)
        )

finally:
    conn.close()
