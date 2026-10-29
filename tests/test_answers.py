from pdfrag.answer import REFUSAL_TEXT, Citation, validate_citations


def test_answer_cites_the_right_page(service):
    res = service.pipeline.ask("What does error E01 mean on the standing desk?")
    assert not res.refused
    assert "motor overload" in res.answer
    assert ("acme-ergodesk-pro-product-guide.pdf", 3) in {(c.source, c.page) for c in res.citations}
    assert "[1]" in res.answer


def test_citations_link_to_the_pdf_page(service):
    res = service.pipeline.ask("How much does white-glove delivery cost?")
    c = res.citations[0]
    assert c.url == f"/pdf/{c.source}#page={c.page}"


def test_cited_snippets_are_flagged(service):
    res = service.pipeline.ask("How much does white-glove delivery cost?")
    cited_ids = {c.chunk_id for c in res.citations}
    assert cited_ids
    assert {s.chunk_id for s in res.snippets if s.cited} == cited_ids


def test_refuses_out_of_scope_question(service):
    res = service.pipeline.ask("What is the warranty on the X500 drone?")
    assert res.refused
    assert res.answer == REFUSAL_TEXT
    assert res.citations == []
    assert res.reason == "low_retrieval_confidence"


def test_refuses_company_question_not_in_docs(service):
    # The company name matches, but nothing says who the CEO is.
    res = service.pipeline.ask("Who is the CEO of Acme Supply Co.?")
    assert res.refused


def test_refuses_empty_or_stopword_question(service):
    for q in ["", "   ", "what is it?"]:
        res = service.pipeline.ask(q)
        assert res.refused and res.reason == "empty_question"


def test_every_answer_has_valid_citations(service, index):
    for q in ["How many days do customers have to return an unopened item?",
              "Who has to approve refunds over $500?",
              "Can furniture be shipped to a PO box?"]:
        res = service.pipeline.ask(q)
        assert not res.refused, q
        assert res.citations
        assert validate_citations(res.citations, index) == []
        retrieved = {s.chunk_id for s in res.snippets}
        for c in res.citations:
            assert c.chunk_id in retrieved  # only retrieved chunks are ever cited


def _real_citation(index):
    chunk = index.chunks[0]
    quote = chunk.page_content.split("\n\n")[-1][:60]
    return Citation(n=1, source=chunk.metadata["source"], page=chunk.metadata["page"],
                    chunk_id=chunk.metadata["chunk_id"], quote=quote, url="")


def test_validation_accepts_a_real_citation(index):
    assert validate_citations([_real_citation(index)], index) == []


def test_validation_rejects_unknown_chunk(index):
    c = _real_citation(index).model_copy(update={"chunk_id": "made-up.pdf#p9-c0"})
    assert validate_citations([c], index)


def test_validation_rejects_wrong_page(index):
    c = _real_citation(index)
    c = c.model_copy(update={"page": c.page + 5})
    assert validate_citations([c], index)


def test_validation_rejects_invented_quote(index):
    c = _real_citation(index).model_copy(update={"quote": "Refunds are issued within 2 hours."})
    assert validate_citations([c], index)
