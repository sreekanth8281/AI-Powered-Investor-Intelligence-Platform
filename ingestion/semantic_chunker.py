import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.embeddings import get_embedding_model


def read_markdown(markdown_file: str) -> str:
    """Read the extracted Markdown document."""
    return Path(markdown_file).read_text(encoding="utf-8")


def is_heading(block: str) -> bool:
    """Check whether a Markdown block is a heading."""
    return bool(re.match(r"^\s{0,3}#{1,6}\s+", block.strip()))


def is_table(block: str) -> bool:
    """Check whether a block contains Markdown table rows."""
    lines = block.strip().splitlines()

    return (
        len(lines) >= 2
        and all(line.strip().startswith("|") for line in lines)
    )


def split_markdown_blocks(markdown: str) -> list[str]:
    """
    Split Markdown into blocks while preserving paragraphs,
    headings, and contiguous table rows.
    """
    lines = markdown.splitlines()
    blocks = []
    current = []
    current_type = None

    def save_current():
        nonlocal current, current_type

        if current:
            content = "\n".join(current).strip()
            if content:
                blocks.append(content)

        current = []
        current_type = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            save_current()
            continue

        if stripped.startswith("|"):
            block_type = "table"
        elif re.match(r"^\s{0,3}#{1,6}\s+", line):
            block_type = "heading"
        else:
            block_type = "text"

        if block_type == "table":
            if current_type != "table":
                save_current()

            current_type = "table"
            current.append(line)

        elif block_type == "heading":
            save_current()
            current_type = "heading"
            current.append(line)

        else:
            if current_type != "text":
                save_current()

            current_type = "text"
            current.append(line)

    save_current()
    return blocks


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens without silently truncating the input."""
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )


def split_table_rows(
    table: str,
    context: str,
    tokenizer,
    max_tokens: int,
    overlap_rows: int = 0,
) -> list[str]:
    """
    Split a long table by rows, preserving the table's
    explanatory context and column headers in each chunk.
    """
    lines = table.splitlines()

    if not lines:
        return []

    # Preserve the first row and Markdown separator as table headers.
    if len(lines) >= 2 and re.search(r"-{3,}", lines[1]):
        table_header = lines[:2]
        data_rows = lines[2:]
    else:
        table_header = lines[:1]
        data_rows = lines[1:]

    prefix_parts = [part for part in [context] if part]
    header_text = "\n".join(table_header)

    chunks = []
    current_rows = []

    def build_chunk(rows):
        parts = prefix_parts + [
            "\n".join(table_header + rows)
        ]
        return "\n\n".join(parts).strip()

    for row in data_rows:
        candidate_rows = current_rows + [row]
        candidate = build_chunk(candidate_rows)

        if count_tokens(candidate, tokenizer) <= max_tokens:
            current_rows.append(row)
            continue

        if current_rows:
            chunks.append(build_chunk(current_rows))

        current_rows = [row]

        # Even a single row may be too large, so handle that case.
        if count_tokens(build_chunk(current_rows), tokenizer) > max_tokens:
            row_text = "\n".join([header_text, row])
            safe_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=max_tokens,
                chunk_overlap=0,
                separators=["\n", " ", ""],
            )

            for part in safe_splitter.split_text(row_text):
                chunks.append(
                    "\n\n".join(prefix_parts + [part]).strip()
                )

            current_rows = []

    if current_rows:
        chunks.append(build_chunk(current_rows))

    return chunks


def chunk_markdown(
    markdown_file: str,
    embeddings=None,
    target_tokens: int = 180,
    chunk_overlap: int = 30,
) -> list[Document]:
    """
    Create context-preserving chunks from Markdown.

    Related headings and explanatory paragraphs are attached
    to their following tables wherever possible.
    """
    markdown = read_markdown(markdown_file)

    if embeddings is None:
        embeddings = get_embedding_model()

    model = embeddings._client
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length

    # Leave room for special tokens.
    safe_limit = min(target_tokens, max_tokens - 2)

    print("Markdown length:", len(markdown))
    print("Embedding model token limit:", max_tokens)

    blocks = split_markdown_blocks(markdown)
    print("Markdown blocks:", len(blocks))

    # Build table chunks with preceding contextual blocks.
    grouped_chunks = []
    pending_context = []
    last_heading = None

    for block in blocks:
        if is_heading(block):
            last_heading = block
            pending_context = [block]
            continue

        if is_table(block):
            context = "\n\n".join(pending_context).strip()

            # Include context with the table if it fits.
            complete = "\n\n".join(
                part for part in [context, block] if part
            )

            if count_tokens(complete, tokenizer) <= safe_limit:
                grouped_chunks.append(complete)
            else:
                table_chunks = split_table_rows(
                    table=block,
                    context=context,
                    tokenizer=tokenizer,
                    max_tokens=safe_limit,
                )
                grouped_chunks.extend(table_chunks)

            pending_context = []
            continue

        # Keep explanatory paragraphs with the current heading.
        if pending_context:
            pending_context.append(block)
        else:
            pending_context = [block]

    # Preserve remaining content that was not followed by a table.
    if pending_context:
        remaining = "\n\n".join(pending_context).strip()

        if remaining:
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=safe_limit,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                add_start_index=True,
            )

            grouped_chunks.extend(splitter.split_text(remaining))

    # Enforce token limits and preserve content that needs splitting.
    final_texts = []

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=safe_limit,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for text in grouped_chunks:
        if count_tokens(text, tokenizer) <= safe_limit:
            final_texts.append(text)
        else:
            final_texts.extend(splitter.split_text(text))

    documents = [
        Document(
            page_content=text,
            metadata={
                "source_file": Path(markdown_file).name,
                "chunk_index": index,
            },
        )
        for index, text in enumerate(final_texts)
    ]

    print("Final chunks:", len(documents))

    token_counts = [
        count_tokens(doc.page_content, tokenizer)
        for doc in documents
    ]

    if token_counts:
        print("Smallest chunk tokens:", min(token_counts))
        print("Largest chunk tokens:", max(token_counts))
        print(
            "Average chunk tokens:",
            round(sum(token_counts) / len(token_counts), 2),
        )

    oversized = [
        i for i, count in enumerate(token_counts)
        if count > max_tokens
    ]
    print("Chunks exceeding model limit:", len(oversized))

    return documents


if __name__ == "__main__":
    markdown_path = "data/markdown/apple_report_2024_fixed.md"

    chunks = chunk_markdown(markdown_path)

    for i, doc in enumerate(chunks):
        text = doc.page_content.lower()

        if (
            "operating expenses for 2024" in text
            or "total operating expenses" in text
        ):
            print(f"\n{'=' * 20} CHUNK {i} {'=' * 20}")
            print(doc.page_content)

