"""Settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = ROOT / "sample_docs"
    top_k: int = 4
    # Minimum query coverage (0..1) of the best retrieved chunk. Below it, refuse.
    min_confidence: float = 0.5
    # Minimum coverage for a single sentence to be used in an extractive answer.
    min_sentence_coverage: float = 0.34
    chunk_size: int = 600
    chunk_overlap: int = 80
    llm_provider: str = "none"   # none | openai | anthropic | gemini | ollama
    llm_model: str = ""
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            docs_dir=Path(os.getenv("DOCS_DIR") or ROOT / "sample_docs"),
            top_k=_int("TOP_K", 4),
            min_confidence=_float("MIN_CONFIDENCE", 0.5),
            min_sentence_coverage=_float("MIN_SENTENCE_COVERAGE", 0.34),
            chunk_size=_int("CHUNK_SIZE", 600),
            chunk_overlap=_int("CHUNK_OVERLAP", 80),
            llm_provider=(os.getenv("LLM_PROVIDER") or "none").strip().lower(),
            llm_model=(os.getenv("LLM_MODEL") or "").strip(),
            ollama_base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip(),
        )
