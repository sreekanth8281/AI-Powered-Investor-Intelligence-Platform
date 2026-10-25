from database.database import get_connection
from psycopg2.extras import Json

def insert(chunks,embeddings,company,year,source_file,metadata=None):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but"
            f"{len(embeddings)} embeddings."
        )
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

            #Remove previous ingestion of this exact document.
            cursor.execute(
                """
                DELETE from document_chunks
                WHERE company = %s
                AND year = %s
                AND source_file = %s
                """,
                (company, year, source_file)
            )

            deleted_count = cursor.rowcount
            print(f"Removed {deleted_count} old chunks.")

            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """insert into document_chunks
                    (chunk,embedding,company, year,source_file,metadata)
                    values(%s,%s,%s,%s,%s,%s)""",
                    (chunk,
                    embedding,
                    company,
                    year,
                    source_file,
                    Json(metadata or {}))
                )

        conn.commit()
        print(f"Inserted {len(chunks)} successfully!")
    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def search(query_embedding,top_k=5):
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
                LIMIT %s;
                """,
                (query_embedding,query_embedding,top_k)
            )

            results = cursor.fetchall()
            return results

    finally:
        conn.close()



def get_chunk_by_ids(chunk_ids):
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
    