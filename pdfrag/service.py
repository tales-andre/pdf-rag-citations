"""Wire settings, ingestion, index and chat model into one pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from .answer import RagPipeline
from .config import Settings
from .ingest import SourceFile, ingest_folder
from .llm import build_chat_model, describe
from .retriever import LexicalIndex


@dataclass
class Service:
    settings: Settings
    pipeline: RagPipeline
    files: list[SourceFile]

    @property
    def chunk_count(self) -> int:
        return len(self.pipeline.index.chunks)


def build_service(settings: Settings | None = None, llm: BaseChatModel | None = None) -> Service:
    """Index the documents folder and return a ready pipeline.

    `llm` overrides the provider from settings (used by tests with a fake model).
    """
    settings = settings or Settings.from_env()
    chunks, files = ingest_folder(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    index = LexicalIndex(chunks)
    if llm is None:
        llm = build_chat_model(settings)
        label = describe(settings)
    else:
        label = f"LLM mode: {type(llm).__name__}"
    return Service(settings=settings, pipeline=RagPipeline(index, settings, llm, label), files=files)
