"""Pluggable chat model. The default is no model at all (extractive mode).

Provider packages are optional (requirements-llm.txt) and imported lazily, so
the offline demo does not need any of them.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings

PROVIDERS = {
    # provider: (python module, class name, pip package)
    "openai": ("langchain_openai", "ChatOpenAI", "langchain-openai"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "langchain-anthropic"),
    "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai"),
    "ollama": ("langchain_ollama", "ChatOllama", "langchain-ollama"),
}


class LLMConfigError(RuntimeError):
    pass


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    """Return a LangChain chat model for LLM_PROVIDER, or None for extractive mode.

    API keys are read by each provider package from its usual variable
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY). Ollama needs no key.
    """
    provider = settings.llm_provider
    if provider in ("", "none", "extractive"):
        return None
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: none, {', '.join(PROVIDERS)}.")
    if not settings.llm_model:
        raise LLMConfigError(f"LLM_PROVIDER={provider} needs LLM_MODEL (see .env.example).")

    module_name, class_name, package = PROVIDERS[provider]
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise LLMConfigError(
            f"Package '{package}' is not installed. Run: pip install -r requirements-llm.txt"
        ) from exc
    cls = getattr(module, class_name)

    kwargs: dict = {"model": settings.llm_model}
    if provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
    if provider != "anthropic":
        # Newer Claude models reject sampling parameters, so leave them at the default there.
        kwargs["temperature"] = 0
    return cls(**kwargs)


def describe(settings: Settings) -> str:
    if settings.llm_provider in ("", "none", "extractive"):
        return "extractive mode (no LLM)"
    return f"LLM mode: {settings.llm_provider} / {settings.llm_model}"
