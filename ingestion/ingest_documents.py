from pathlib import Path

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from ingestion.embeddings import get_embedding_model

from database.vector_store import insert


def ingest_document(
        pdf_file: str,
        company: str,
        year: int
):

    #Pdf markdown
    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        path_pdf=pdf_file,
        output_dir="data/markdown"
    )

    print(f"Markdown created: {markdown_file}")

    #Load embedding model.
    embedding_model = get_embedding_model()


    #markdown -> semantic chunker.
    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embedding_model
    )

    print(f"Created {len(chunks)} semantic chunks.")


    #Extract text from Document objects.
    texts = [chunk.page_content for chunk in chunks]


    #chunks -> embedding vectors
    embeddings = embedding_model.embed_documents(texts)

    #insert into Postgresql
    source_file = Path(pdf_file).name

    insert(
        chunks=texts,
        embeddings=embeddings,
        company=company,
        year=year,
        source_file=source_file,
        metadata={
            "document_type": "annual-report"
        }
    )

    print(f"Document ingestion completed successfully!")

print("ingest_documents.py started")

if __name__ == "__main__":
    ingest_document(
        pdf_file="data/raw_pdfs/apple_report_2024.pdf",
        company="Apple",
        year=2024
    )



     


