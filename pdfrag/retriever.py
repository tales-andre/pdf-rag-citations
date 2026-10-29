"""Offline retrieval: BM25 ranking plus a confidence score the bot can refuse on.

BM25 scores are not comparable across questions, so they make a poor refusal
threshold. Confidence here is *query coverage*: the share of the question's
informative words (weighted by rarity, IDF) that appear in a retrieved chunk.
Words that never appear in the documents get the highest weight, so a question
about something the documents do not mention ("X500 drone") gets low
confidence even when common words ("warranty") match.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from .text import query_groups, tokenize

Groups = list[frozenset[str]]


def as_groups(terms: Iterable[str] | Groups) -> Groups:
    return [t if isinstance(t, frozenset) else frozenset([t]) for t in terms]


def weighted_coverage(groups: Groups, text_terms: set[str], weight: Callable[[str], float]) -> float:
    """Share of the question's weight found in a text (0..1).

    A group (a multi-word name) weighs as much as its rarest word and gets
    partial credit for the words that are present.
    """
    total = found = 0.0
    for group in groups:
        w = max(weight(t) for t in group)
        total += w
        found += w * len(group & text_terms) / len(group)
    return found / total if total else 0.0


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float      # BM25 score, used for ranking only
    coverage: float   # 0..1, IDF-weighted share of question terms found in the chunk


class LexicalIndex:
    """BM25 index over chunks, with IDF weights for coverage scoring."""

    def __init__(self, chunks: list[Document]):
        if not chunks:
            raise ValueError("No chunks to index. Is the documents folder empty?")
        self.chunks = chunks
        self.by_id = {c.metadata["chunk_id"]: c for c in chunks}
        self._tokens = [tokenize(c.page_content) for c in chunks]
        self._term_sets = [set(t) for t in self._tokens]
        self._bm25 = BM25Okapi(self._tokens)
        self._df = Counter(term for terms in self._term_sets for term in terms)
        self._n = len(chunks)

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def coverage(self, query: Iterable[str] | Groups, text_terms: set[str]) -> float:
        return weighted_coverage(as_groups(query), text_terms, self.idf)

    def search(self, question: str, k: int = 4) -> list[Hit]:
        query_terms = tokenize(question)
        if not query_terms:
            return []
        scores = self._bm25.get_scores(query_terms)
        ranked = sorted(range(self._n), key=lambda i: scores[i], reverse=True)
        groups = query_groups(question)
        hits = []
        for i in ranked[:k]:
            cov = self.coverage(groups, self._term_sets[i])
            if cov == 0:
                continue
            hits.append(Hit(doc=self.chunks[i], score=float(scores[i]), coverage=cov))
        return hits


class BM25PageRetriever(BaseRetriever):
    """LangChain retriever wrapper so the index plugs into any LangChain chain.

    Scores are copied into each returned Document's metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: LexicalIndex
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        out = []
        for hit in self.index.search(query, self.k):
            meta = dict(hit.doc.metadata, score=round(hit.score, 4), coverage=round(hit.coverage, 4))
            out.append(Document(page_content=hit.doc.page_content, metadata=meta))
        return out
