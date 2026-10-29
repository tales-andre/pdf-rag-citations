FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# WITH_LLM=1 also installs the optional provider packages (OpenAI, Anthropic, Gemini, Ollama).
ARG WITH_LLM=0
COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_LLM" = "1" ]; then pip install --no-cache-dir -r requirements-llm.txt; fi

COPY pdfrag ./pdfrag
COPY eval ./eval
COPY sample_docs ./sample_docs

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "pdfrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
