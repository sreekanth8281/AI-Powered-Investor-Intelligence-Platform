from ingestion.embeddings import get_embedding_model
from database.vector_store import search



def retrieve(query: str, top_k: int=5) :

    embedding_model = get_embedding_model()

    query_embedding = embedding_model.embed_query(query)

    results = search(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return results

if __name__ == "__main__":
    query = "what was Apple's revenue in 2024?"

    results = retrieve(query,top_k=5)

    print(f"\nQuery:{query}")
    print(f"Retrieved {len(results)} chunks\n")

    for result in results:
        print("ID:", result[0])
        print("Chunk:", result[1][:500])
        print("Company:", result[2])
        print("Year:", result[3])
        print("Distance:", result[-1])
        print("-" * 80)
