import re
from collections import Counter
import math
from database.database import get_connection

class BM25:
    STOP_WORDS = {
        "what", "was", "the", "a", "an",
        "is", "are", "in", "of", "to",
        "for", "and", "or", "on", "with"
    }
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        #Load chunks from PostgreSQL
        self.rows = self.get_all_chunks()
        self.documents = [row[1] for row in self.rows]

        #calculate BM25 information once during initialization
        self.doc_freq = self.document_frequency(self.documents)
        self.idf = self.inverse_document_frequency(
            len(self.documents), self.doc_freq
        )

        self.lengths = self.document_lengths(self.documents)
        self.avgdl = self.average_document_length(self.lengths)


    @classmethod
    def tokenize(cls, text):
        text = text.lower()
        text = re.sub(r"'s\b", "", text)

        tokens = re.findall(r"\b\w+\b", text)

        return [
            token for token in tokens
            if token not in cls.STOP_WORDS
        ]

        

    @staticmethod
    def term_frequency(tokens):
        return Counter(tokens)

    @classmethod
    def document_frequency(cls, documents):
        df = Counter()

        for document in documents:
            tokens = set(cls.tokenize(document))
            df.update(tokens)

        return df

    @staticmethod
    def inverse_document_frequency(totals_documents, df):
        idf = {}
        for term, frequency in df.items():
            idf[term] = math.log(
                1 + (totals_documents - frequency + 0.5) 
                / (frequency + 0.5)
            )
        return idf

    @classmethod
    def document_lengths(cls, documents):
        length = []
        for document in documents:
            tokens = cls.tokenize(document)
            length.append((len(tokens)))
        return length

    @staticmethod
    def average_document_length(lengths):
        if not lengths:
            return 0
        
        return sum(lengths) / len(lengths)


    def bm25(self, tf, idf, doc_length, avgdl):
        if avgdl == 0:
            return 0
        
        numerator = tf * (self.k1 + 1)

        denominator = (tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl)))

        return idf * (numerator / denominator)

    def bm25_score(self, query, document):
        query_tokens = self.tokenize(query)
        document_tokens = self.tokenize(document)

        tf = self.term_frequency(document_tokens)
        doc_length = len(document_tokens)

        score = 0
        for term in query_tokens:
            if term in self.idf:
                score += self.bm25(
                    tf[term],
                    self.idf[term],
                    doc_length,
                    self.avgdl
                )

        return score

    @staticmethod
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


    def search(self, query, top_k= 5):
        res = []

        for row in self.rows:
            chunk_id , document = row[0], row[1]
            score = self.bm25_score(query, document)

            res.append((chunk_id, score, document))

        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_k]


if __name__ == "__main__":
    bm25 = BM25() 
    print("Total chunks loaded:", len(bm25.rows)) 
    print("Average document length:", bm25.avgdl) 
    print("Unique terms in index:", len(bm25.idf)) 
    query = "What were Apple's operating expenses in 2024?" 
    results = bm25.search(query, top_k=5) 
    print("\nBM25 Search Results:") 
    for rank, (chunk_id, score, document) in enumerate(results, start=1): 
        print(f"\nRank: {rank}") 
        print(f"Chunk ID: {chunk_id}") 
        print(f"BM25 Score: {score:.4f}") 
        print(f"Chunk Preview: {document[:500]}")

