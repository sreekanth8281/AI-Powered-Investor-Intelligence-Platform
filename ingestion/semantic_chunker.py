from pathlib import Path
from ingestion.embeddings import get_embedding_model

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter




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
    

    semantic_splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_type = 'percentile'
    )

    #Identify semantic boundaries
    semantic_chunks = semantic_splitter.create_documents(
        [markdown_content]
    )
    print(f"Semantic chunks: {len(semantic_chunks)}")

    #split oversized semantic chunks.
    model = embeddings._client
    max_tokens = model.max_seq_length
    target_tokens = 180

    #keep safety margin for special tokens.
    if target_tokens >= max_tokens:
        target_tokens = max_tokens - 20

    final_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        model.tokenizer,
        chunk_size=target_tokens,
        chunk_overlap=30,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True
    )

    final_chunks = final_splitter.split_documents(semantic_chunks)

    print("Final chunks:", len(final_chunks))

    token_counts = [
        len(
            model.tokenizer(
                doc.page_content,
                add_special_tokens = True,
                truncation=False,
            )["input_ids"]
        )
        for doc in final_chunks
    ]

    print(f"Smallest chunk tokens: {min(token_counts)}")
    print(f"Largest chunk tokens: {max(token_counts)}")
    print(f"Average chunk tokens: {round(sum(token_counts) / len(token_counts))}")


    oversized = [i for i, count in enumerate(token_counts)
                 if count > max_tokens]
    print(f"chunks exceeding model limit: {len(oversized)}")

    if oversized:
        print("warning.Some chunks still exceed the model limit.")

    return final_chunks

if __name__ == "__main__":
    import os
    embeddings = get_embedding_model()

    markdown_file = "data/markdown/apple_report_2024_fixed.md"
    
    chunks = chunk_markdown(
        markdown_file = markdown_file,
        embeddings = embeddings
    )

    print(f"Created {len(chunks)} final chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nchunk {i}")
        print(f"characters: {len(chunk.page_content)}")
        print(chunk.page_content[:300])










   