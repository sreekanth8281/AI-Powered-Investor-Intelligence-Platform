from pathlib import Path
from ingestion.embeddings import get_embedding_model

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker




def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")

def chunk_markdown(
        markdown_file: str,
        embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown

    Args:
       markdown_file: Markdown file path
       embeddings: Embedding model used to calculate semantic similarity.

    Returns:
        List of semantic chunks.
    """
    markdown_content = read_markdown(markdown_file)

    print("Markdown length:", len(markdown_content))
    print("Markdown preview:")
    print(markdown_content[:500])

    splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_type = 'percentile'
    )
    return splitter.create_documents([markdown_content])


if __name__ == "__main__":
    import os
    embeddings = get_embedding_model()

    markdown_file = "data/markdown/apple_report_2024.md"
    
    chunks = chunk_markdown(
        markdown_file = markdown_file,
        embeddings = embeddings
    )

    print(f"Created {len(chunks)} chunks")

   