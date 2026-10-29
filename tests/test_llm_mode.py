"""LLM mode with a fake model: the model's output is checked before it is shown."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from pdfrag.answer import RagPipeline
from pdfrag.config import Settings
from pdfrag.llm import LLMConfigError, build_chat_model

Q = "After the warehouse inspects a return, when is the refund issued?"


def _pipeline(index, settings, reply: str) -> RagPipeline:
    return RagPipeline(index, settings, FakeListChatModel(responses=[reply]), "LLM mode: fake")


def _source_number(index, settings, source, page) -> int:
    """The [S#] number the model would see for a given file and page."""
    docs = _pipeline(index, settings, "").retriever.invoke(Q)
    for i, d in enumerate(docs, start=1):
        if (d.metadata["source"], d.metadata["page"]) == (source, page):
            return i
    raise AssertionError("expected page was not retrieved")


def test_grounded_llm_answer_is_shown(index, settings):
    n = _source_number(index, settings, "acme-returns-and-refunds-sop.pdf", 2)
    reply = f"Refunds are issued to the original payment method within 5 business days after the warehouse inspection [S{n}]."
    res = _pipeline(index, settings, reply).ask(Q)
    assert not res.refused, res.reason
    assert res.answer.endswith("[1].")
    assert (res.citations[0].source, res.citations[0].page) == ("acme-returns-and-refunds-sop.pdf", 2)


def test_llm_not_found_becomes_refusal(index, settings):
    res = _pipeline(index, settings, "NOT_FOUND").ask(Q)
    assert res.refused and res.reason == "llm_not_found"


def test_llm_citing_a_source_that_was_not_retrieved_is_refused(index, settings):
    res = _pipeline(index, settings, "Refunds are issued within 5 business days [S99].").ask(Q)
    assert res.refused and res.reason == "invalid_citation"


def test_llm_answer_without_citations_is_refused(index, settings):
    res = _pipeline(index, settings, "Refunds are issued within 5 business days.").ask(Q)
    assert res.refused and res.reason == "no_citations"


def test_llm_number_not_in_source_is_refused(index, settings):
    n = _source_number(index, settings, "acme-returns-and-refunds-sop.pdf", 2)
    res = _pipeline(index, settings, f"Refunds are issued within 30 business days after inspection [S{n}].").ask(Q)
    assert res.refused and res.reason == "ungrounded_number"


def test_llm_uncited_extra_claim_is_refused(index, settings):
    n = _source_number(index, settings, "acme-returns-and-refunds-sop.pdf", 2)
    reply = (f"Refunds are issued within 5 business days after the warehouse inspection [S{n}]. "
             "Customers also get a discount coupon for their next order.")
    res = _pipeline(index, settings, reply).ask(Q)
    assert res.refused and res.reason == "uncited_statement"


def test_llm_statement_unrelated_to_cited_source_is_refused(index, settings):
    n = _source_number(index, settings, "acme-returns-and-refunds-sop.pdf", 2)
    reply = f"Customers receive a complimentary gift voucher and loyalty points [S{n}]."
    res = _pipeline(index, settings, reply).ask(Q)
    assert res.refused and res.reason == "ungrounded_statement"


def test_low_confidence_question_never_reaches_the_model(index, settings):
    class Boom(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise AssertionError("model must not be called")

    res = RagPipeline(index, settings, Boom(responses=["x"]), "LLM mode: fake").ask("Can I pay with cryptocurrency?")
    assert res.refused and res.reason == "low_retrieval_confidence"


def test_model_error_is_reported_not_guessed(index, settings):
    class Broken(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("provider down")

    res = RagPipeline(index, settings, Broken(responses=["x"]), "LLM mode: fake").ask(Q)
    assert res.refused and res.reason == "llm_error"
    assert "Nothing was guessed" in res.answer


def test_default_provider_is_extractive():
    assert build_chat_model(Settings()) is None


def test_provider_needs_a_model_name():
    with pytest.raises(LLMConfigError):
        build_chat_model(Settings(llm_provider="openai", llm_model=""))


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMConfigError):
        build_chat_model(Settings(llm_provider="acme-llm", llm_model="x"))


@pytest.mark.parametrize("provider,module,env", [
    ("openai", "langchain_openai", "OPENAI_API_KEY"),
    ("anthropic", "langchain_anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "langchain_google_genai", "GOOGLE_API_KEY"),
    ("ollama", "langchain_ollama", None),
])
def test_provider_models_can_be_built(provider, module, env, monkeypatch):
    """Builds each provider's LangChain chat model with a dummy key. No network call is made."""
    pytest.importorskip(module)
    if env:
        monkeypatch.setenv(env, "dummy-key-for-test")
    model = build_chat_model(Settings(llm_provider=provider, llm_model="some-model"))
    assert model is not None
