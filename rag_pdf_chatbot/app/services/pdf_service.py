"""PDF text extraction and chunking."""

from pypdf import PdfReader


class PDFService:
    """Handles extracting text from PDFs and splitting it into chunks.

    Kept separate from embedding/storage/LLM concerns (Single Responsibility)
    so the chunking strategy can change independently of everything else.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize with the desired chunk size and overlap, in characters."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str) -> str:
        """Extract and concatenate text from every page of a PDF."""
        reader = PdfReader(file_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping fixed-size character chunks."""
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self._chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_length:
                break
            start = end - self._chunk_overlap

        return chunks
