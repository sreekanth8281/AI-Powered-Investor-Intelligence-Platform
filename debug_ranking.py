from ingestion.embeddings import get_embedding_model
from database.vector_store import get_connection

embedding_model = get_embedding_model()

query = "What were Apple's operating expenses in 2024?"
query_embedding = embedding_model.embed_query(query)

conn = get_connection()

try:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, chunk, company, year,
                   embedding <=> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 10;
            """,
            (query_embedding, query_embedding),
        )

        results = cursor.fetchall()

    for rank, row in enumerate(results, start=1):
        chunk_id, chunk, company, year, distance = row

        print(f"\nRank: {rank} | ID: {chunk_id} | Distance: {distance:.6f}")
        print(f"Company: {company} | Year: {year}")
        print("Chunk text:")
        print(chunk[:1000])
        print("-" * 80)

finally:
    conn.close()