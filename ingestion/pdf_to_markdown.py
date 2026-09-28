from pathlib import Path

import pymupdf4llm

class PDFToMarkdownConverter:
    """Convert pdf documents to Markdown"""

    def convert_pdf(self, path_pdf: str, output_dir: str) ->str:
        """
        Convert a PDF document to Markdown.

        Args:
            path: Source PDf path.
            output_dir: output markdown directory.

        Returns:
             Generated markdown files.
        """
        pdf_file = Path(path_pdf)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDf file not found:{pdf_file}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        markdown_content = pymupdf4llm.to_markdown(str(pdf_file))

        print("Extracted markdown length:", len(markdown_content))
        print("Extracted markdown preview:", repr(markdown_content[:500]))          

        markdown_file = output_path / f"{pdf_file.stem}.md"

        markdown_file.write_text(
            markdown_content,
            encoding="utf-8"
        )
        return str(markdown_file)

    def convert_directory(self, input_dir: str, output_dir: str) -> list[str]:
        """
        Convert all PDFs from a directory to Markdown.
        
        Args:
           input_dir: directory containing PDf files.
           output_dir: Directory to save markdown files.

        Returns:
             List of generated markdown files.
        """
        input_path = Path(input_dir)
        markdown_files = []

        for pdf_file in input_path.glob("*.pdf"):
            markdown_file = self.convert_pdf(
                path_pdf = str(pdf_file),
                output_dir= output_dir
            )
            markdown_files.append(markdown_file)

        return markdown_files

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    input_dir = repo_root / "data" / "raw_pdfs"
    output_dir = repo_root / "data" / "markdown"

    converter = PDFToMarkdownConverter()

    markdown_files = converter.convert_directory(
        input_dir = str(input_dir),
        output_dir = str(output_dir)
    )

    print(f"Successfully converted {len(markdown_files)} PDF(s):")



