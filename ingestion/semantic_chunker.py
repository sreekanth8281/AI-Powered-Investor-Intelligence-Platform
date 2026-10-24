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

    markdown_file = "data/markdown/apple_report_2024_fixed.md"
    
    chunks = chunk_markdown(
        markdown_file = markdown_file,
        embeddings = embeddings
    )

    print(f"Created {len(chunks)} chunks")


    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print(f"smallest chunk {min(chunk_lengths)} characters")
    print(f"largest chunk {max(chunk_lengths)} characters")
    print(f"average chunk {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")


    print("Maximum sequence length:", embeddings._client.max_seq_length)

    largest_chunk_id, largest_chunk = max(
    enumerate(chunks),
    key=lambda item: len(item[1].page_content)
)

    print("Largest chunk ID:", largest_chunk_id)
    print("Character count:", len(largest_chunk.page_content))
    print("Beginning:", largest_chunk.page_content[:500])
    print("Ending:", largest_chunk.page_content[-500:])


    model = embeddings._client
    max_length = model.max_seq_length

    print(f"max seq length:", max_length)

    for i in [24,46,64]:
        if i < len(chunks):
            text = chunks[i].page_content

            token_ids = model.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]

            print(f"\nchunk {i}")
            print(f"characters: {len(text)}")
            print(f"token_count: {len(token_ids)}")
            print("Exceeds limit:", len(token_ids) > max_length)






   