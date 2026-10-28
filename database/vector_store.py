from database.database import get_connection
from psycopg2.extras import Json


def insert(chunks, embeddings, company, year, source_file, metadata=None):
    """
    Insert unique chunks and their corresponding embeddings into PostgreSQL.

    Duplicate chunks within the current ingestion batch are removed before
    insertion. Previous rows for the same company, year, and source file
    are deleted before the new chunks are inserted.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but "
            f"{len(embeddings)} embeddings."
        )

    # Step 1: Remove duplicate chunks while preserving embedding alignment.
    unique_chunks = []
    unique_embeddings = []
    seen_chunks = set()

    for chunk, embedding in zip(chunks, embeddings):
        # Normalize whitespace to identify equivalent chunk text.
        normalized_chunk = " ".join(chunk.split())

        if normalized_chunk not in seen_chunks:
            seen_chunks.add(normalized_chunk)
            unique_chunks.append(chunk)
            unique_embeddings.append(embedding)

    duplicates_removed = len(chunks) - len(unique_chunks)

    print(f"Original chunks: {len(chunks)}")
    print(f"Unique chunks: {len(unique_chunks)}")
    print(f"Duplicate chunks removed: {duplicates_removed}")

    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            # Step 2: Remove previous ingestion of this exact document.
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE company = %s
                  AND year = %s
                  AND source_file = %s
                """,
                (company, year, source_file)
            )

            deleted_count = cursor.rowcount
            print(f"Removed {deleted_count} old chunks.")

            # Step 3: Insert only unique chunks and their embeddings.
            for chunk, embedding in zip(
                unique_chunks, unique_embeddings
            ):
                cursor.execute(
                    """
                    INSERT INTO document_chunks
                    (
                        chunk,
                        embedding,
                        company,
                        year,
                        source_file,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk,
                        embedding,
                        company,
                        year,
                        source_file,
                        Json(metadata or {})
                    )
                )

        # Step 4: Commit the deletion and insertion together.
        conn.commit()

        print(
            f"Inserted {len(unique_chunks)} unique chunks successfully!"
        )

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def search(query_embedding, top_k=5):
    """
    Retrieve the top-k chunks using cosine distance.

    Lower distance means greater cosine similarity.
    """

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    chunk,
                    company,
                    year,
                    source_file,
                    metadata,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k)
            )

            return cursor.fetchall()

    finally:
        conn.close()


def get_chunk_by_ids(chunk_ids):
    """
    Retrieve database chunks using their IDs.
    """

    if not chunk_ids:
        return []

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    chunk,
                    company,
                    year,
                    source_file,
                    metadata
                FROM document_chunks
                WHERE id = ANY(%s)
                """,
                (chunk_ids,)
            )

            return cursor.fetchall()

    finally:
        conn.close()

