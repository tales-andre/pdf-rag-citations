"""PDF ingestion: one LangChain Document per page, then page-bound chunks.

Every chunk carries `source` (file name), `page` (1-based, as printed in a PDF
viewer) and a stable `chunk_id`. Chunks never cross a page boundary, so a
citation always points to exactly one page.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .text import reflow_page_text

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    name: str
    path: Path
    pages: int


def _boilerplate_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop header/footer lines that repeat on most pages of the same file."""
    if len(pages) < 2:
        return pages
    counts = Counter()
    for text in pages:
        counts.update({_boilerplate_key(line) for line in text.splitlines() if line.strip()})
    threshold = max(2, int(len(pages) * 0.6 + 0.5))
    repeated = {key for key, n in counts.items() if n >= threshold}
    return [
        "\n".join(line for line in text.splitlines() if _boilerplate_key(line) not in repeated)
        for text in pages
    ]


def load_pdf_pages(path: Path) -> tuple[list[Document], int]:
    """Read a PDF. Returns one Document per page with text, and the page count."""
    reader = PdfReader(str(path))
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    cleaned = _strip_repeated_lines(raw_pages)
    docs = []
    for number, text in enumerate(cleaned, start=1):
        text = reflow_page_text(text)
        if not text.strip():
            log.warning("No extractable text on %s page %s (scanned image?)", path.name, number)
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name, "page": number}))
    return docs, len(raw_pages)


def split_pages(pages: list[Document], chunk_size: int = 600, chunk_overlap: int = 80) -> list[Document]:
    """Split each page on its own so chunks keep a single, exact page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    chunks: list[Document] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.page_content)):
            meta = dict(page.metadata)
            meta["chunk_id"] = f"{meta['source']}#p{meta['page']}-c{i}"
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def ingest_folder(folder: Path, chunk_size: int = 600, chunk_overlap: int = 80) -> tuple[list[Document], list[SourceFile]]:
    """Load every PDF in a folder (not recursive). Returns chunks and the file list."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")
    chunks: list[Document] = []
    files: list[SourceFile] = []
    for path in sorted(folder.glob("*.pdf")):
        try:
            pages, page_count = load_pdf_pages(path)
        except Exception as exc:  # a broken PDF should not take the service down
            log.error("Skipping %s: %s", path.name, exc)
            continue
        files.append(SourceFile(name=path.name, path=path.resolve(), pages=page_count))
        chunks.extend(split_pages(pages, chunk_size, chunk_overlap))
    log.info("Indexed %d chunks from %d PDF files in %s", len(chunks), len(files), folder)
    return chunks, files
