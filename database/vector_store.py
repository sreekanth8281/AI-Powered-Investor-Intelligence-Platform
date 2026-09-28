from database.database import get_connection
from psycopg2.extras import Json

def insert(chunks,embeddings,company,year,source_file,metadata=None):
    conn = get_connection()

    try:
        with conn.cursor() as cursor:

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
        conn.rollback
        raise

    finally:
        conn.close()
    