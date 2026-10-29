from pypdf import PdfReader

from pdfrag.config import ROOT
from pdfrag.ingest import ingest_folder, load_pdf_pages, split_pages

DOCS = ROOT / "sample_docs"


def test_every_chunk_has_file_page_and_id(index):
    for chunk in index.chunks:
        meta = chunk.metadata
        assert meta["source"].endswith(".pdf")
        assert isinstance(meta["page"], int) and meta["page"] >= 1
        assert meta["chunk_id"].startswith(f"{meta['source']}#p{meta['page']}-")


def test_chunk_ids_are_unique(index):
    ids = [c.metadata["chunk_id"] for c in index.chunks]
    assert len(ids) == len(set(ids))


def test_page_numbers_match_the_pdf():
    """A fact printed on page 3 of the PDF must be indexed as page 3."""
    path = DOCS / "acme-ergodesk-pro-product-guide.pdf"
    pages, count = load_pdf_pages(path)
    assert count == len(PdfReader(str(path)).pages) == 3
    by_page = {p.metadata["page"]: p.page_content for p in pages}
    assert "E01 means motor overload" in by_page[3]
    assert "E01" not in by_page[1] and "E01" not in by_page[2]


def test_chunks_never_cross_pages():
    pages, _ = load_pdf_pages(DOCS / "acme-returns-and-refunds-sop.pdf")
    chunks = split_pages(pages, chunk_size=200, chunk_overlap=20)
    page_text = {p.metadata["page"]: p.page_content for p in pages}
    assert len(chunks) > len(pages)  # a small chunk size forces several chunks per page
    for c in chunks:
        assert c.page_content in page_text[c.metadata["page"]]


def test_repeated_footer_is_removed():
    pages, _ = load_pdf_pages(DOCS / "acme-shipping-faq.pdf")
    for p in pages:
        assert "fictional company) -" not in p.page_content


def test_all_sample_files_are_listed(service):
    names = {f.name for f in service.files}
    assert names == {p.name for p in DOCS.glob("*.pdf")}
    assert all(f.pages >= 2 for f in service.files)


def test_broken_pdf_is_skipped(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "good.pdf").write_bytes((DOCS / "acme-warranty-policy.pdf").read_bytes())
    chunks, files = ingest_folder(tmp_path)
    assert [f.name for f in files] == ["good.pdf"]
    assert chunks and all(c.metadata["source"] == "good.pdf" for c in chunks)
