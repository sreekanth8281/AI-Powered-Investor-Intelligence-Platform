import re
from collections import Counter
import math
from database.database import get_connection

STOP_WORDS = {
    "what", "was", "the", "a", "an",
    "is", "are", "in", "of", "to",
    "for", "and", "or", "on", "with"
}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)

    words = re.findall(r"\b\w+\b", text)

    tokens = []

    for word in words:
        if word not in STOP_WORDS:
            tokens.append(word)

    return tokens

def term_frequency(tokens):
    return Counter(tokens)

def document_frequency(documents):
    df = Counter()

    for document in documents:
        tokens = set(tokenize(document))

        for token in tokens:
            df[token] += 1

    return df

def inverse_document_frequency(totals_documents, df):
    idf = {}
    for term, frequency in df.items():
        idf[term] = math.log(
            1 + (totals_documents - frequency + 0.5) 
            / (frequency + 0.5)
        )
    return idf

def document_lengths(documents):
    length = []
    for document in documents:
        tokens = tokenize(document)
        length.append((len(tokens)))
    return length


def average_document_length(lengths):
    return sum(lengths) / len(lengths)


def bm25(tf, idf, doc_length, avgdl, k1=1.5, b=0.75):
    numerator = tf * (k1 + 1)

    denominator = (tf + k1 * (1 - b + b * (doc_length / avgdl)))

    return idf * (numerator / denominator)

def bm25_score(query, document, idf, avgdl, k1=1.5, b=0.75):
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    tf = term_frequency(document_tokens)
    doc_length = len(document_tokens)

    score = 0
    for term in query_tokens:
        if term not in tf:
            continue

        term_score = bm25(
            tf= tf[term],
            idf=idf[term],
            doc_length=doc_length,
            avgdl=avgdl,
            k1=k1,
            b=b
        )

        score += term_score

    return score


def get_all_chunks():
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, chunk
                FROM document_chunks
                ORDER BY id;
                """)
            return cursor.fetchall()

    finally:
        conn.close()


def bm25_search(query, rows, idf, avgdl, top_k= 5):
    res = []

    for chunk_id, document in rows:
        score = bm25_score(
            query=query,
            document=document,
            idf=idf,
            avgdl=avgdl
        )
        res.append((chunk_id, score, document))

    res.sort(key=lambda x: x[1], reverse=True)
    return res[:top_k]


if __name__ == "__main__":

    rows = get_all_chunks()

    documents = [row[1] for row in rows]

    print("Number of documents for BM25:", len(documents))

    df = document_frequency(documents)

    idf = inverse_document_frequency(
        len(documents),
        df
    )

    lengths = document_lengths(documents)

    avgdl = average_document_length(lengths)

    print("Number of documents:", len(documents))
    print("Average document length:", avgdl)

    print("Number of chunks:", len(rows))
    print("First chunk ID:", rows[0][0])
    print("First chunk:", rows[0][1][:300])



    query = "What was Apple's net sales in 2024?"

    results = bm25_search(
        query=query,
        rows=rows,
        idf=idf,
        avgdl=avgdl,
        top_k=5
    )

    print("\nQuery:", query)
    print("\nBM25 Top 5:")

    for rank, (chunk_id, score, document) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Chunk ID:", chunk_id)
        print("BM25 Score:", score)
        print("Chunk:", document[:500])