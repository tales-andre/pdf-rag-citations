"""Small text helpers shared by ingestion, retrieval and answering.

Everything here is deterministic, so the whole pipeline can run offline and
be tested without a model.
"""

from __future__ import annotations

import re

import snowballstemmer

# Function words and question phrasing. They carry no evidence about *where*
# an answer lives, so they are ignored for retrieval and confidence scoring.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers him his how i if
    in into is it its itself just me more most my myself no nor not of off on once
    only or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you
    your yours yourself yourselves ok okay please tell know want need get got
    long much many anything something someone way
    locate located location find exactly
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_WS_RE = re.compile(r"\s+")
_RAW_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")


_STEMMER = snowballstemmer.stemmer("english")


def stem(word: str) -> str:
    """Snowball (Porter 2) stemming so 'returns', 'returned' and 'return' match."""
    return word if word[0].isdigit() else _STEMMER.stemWord(word)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split into word/number tokens, drop stopwords, stem."""
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if not keep_stopwords and raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    return tokens


def query_groups(text: str) -> list[frozenset[str]]:
    """Question terms for confidence scoring, with multi-word names counted once.

    "Who is the CEO of Acme Supply Co.?" gives {ceo} and {acme, suppli, co}:
    a company or product name is one concept, so matching it should not
    outweigh the one word the question is really about.
    """
    groups: list[frozenset[str]] = []
    run: list[str] = []

    def flush():
        if len(run) >= 2:
            groups.append(frozenset(run))
        else:
            groups.extend(frozenset([t]) for t in run)
        run.clear()

    for raw in _RAW_WORD_RE.findall(text):
        low = raw.lower()
        if low in STOPWORDS:
            flush()
            continue
        term = stem(low)
        if raw[0].isupper():
            run.append(term)
        else:
            flush()
            groups.append(frozenset([term]))
    flush()
    return list(dict.fromkeys(groups))


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def reflow_page_text(raw: str) -> str:
    """Turn PDF-extracted lines into paragraphs.

    PDF text comes back with a newline at every wrapped line. Short lines with
    no closing punctuation are treated as headings and kept as their own
    paragraph; other lines are joined with the next one.
    """
    lines = [normalize_ws(line) for line in raw.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        is_heading = len(line) < 60 and not line.endswith((".", ",", ";", ":", "?", "!", ")"))
        if is_heading:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
            continue
        current.append(line)
        if line.endswith((".", "?", "!", ":")) and len(line) < 70:
            # A short line ending a sentence is usually the end of a paragraph.
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> list[str]:
    """Split a chunk into sentences; paragraph breaks always end a sentence."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_ws(paragraph)
        if not paragraph:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip())
    return sentences


def numbers_in(text: str) -> set[str]:
    """Numeric facts (prices, days, percentages) mentioned in a text."""
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))
