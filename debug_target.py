from database.vector_store import get_connection

chunk_ids = [279, 307, 393, 529]

conn = get_connection()

try:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, chunk
            FROM document_chunks
            WHERE id = ANY(%s)
            ORDER BY id;
            """,
            (chunk_ids,),
        )

        results = cursor.fetchall()

        for chunk_id, chunk in results:
            print(f"\n{'=' * 30} CHUNK {chunk_id} {'=' * 30}")
            print(chunk)
            print("=" * 80)

finally:
    conn.close()