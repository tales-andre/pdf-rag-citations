import pytest
from fastapi.testclient import TestClient

from pdfrag.api import create_app


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health_reports_extractive_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "extractive mode (no LLM)"
    assert body["documents"] == 5


def test_chat_answers_with_citations(client):
    r = client.post("/chat", json={"question": "How much does white-glove delivery cost?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert "$149" in body["answer"]
    c = body["citations"][0]
    assert c["source"] == "acme-shipping-faq.pdf" and c["page"] == 2
    assert body["snippets"]


def test_chat_refuses_out_of_scope(client):
    body = client.post("/chat", json={"question": "Do you have a retail store in Chicago?"}).json()
    assert body["refused"] is True
    assert body["answer"] == "I couldn't find this in the documents."
    assert body["citations"] == []


def test_chat_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_citation_link_serves_the_pdf(client):
    body = client.post("/chat", json={"question": "Where is the serial number located?"}).json()
    url = body["citations"][0]["url"].split("#")[0]
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


@pytest.mark.parametrize("name", ["..%2FREADME.md", "missing.pdf", "..%5Cpyproject.toml", "README.md"])
def test_pdf_route_only_serves_indexed_files(client, name):
    assert client.get(f"/pdf/{name}").status_code == 404


def test_documents_list(client):
    docs = client.get("/api/documents").json()
    assert {d["name"] for d in docs} >= {"acme-shipping-faq.pdf", "acme-warranty-policy.pdf"}


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
