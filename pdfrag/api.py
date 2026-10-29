"""FastAPI app: /chat, a document list, the PDFs themselves and the web UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import ChatResult
from .service import Service, build_service

STATIC = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(service: Service | None = None) -> FastAPI:
    state: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["service"] = service or build_service()
        yield

    app = FastAPI(title="PDF answers with citations", version="1.0.0", lifespan=lifespan)

    def svc() -> Service:
        return state["service"]

    @app.get("/health")
    def health():
        s = svc()
        return {"status": "ok", "mode": s.pipeline.mode, "documents": len(s.files), "chunks": s.chunk_count}

    @app.get("/api/documents")
    def documents():
        return [{"name": f.name, "pages": f.pages} for f in svc().files]

    @app.post("/chat", response_model=ChatResult)
    def chat(req: ChatRequest) -> ChatResult:
        return svc().pipeline.ask(req.question)

    @app.get("/pdf/{name}")
    def pdf(name: str):
        # Only files that were indexed can be opened; the name is looked up, never joined to a path.
        match = next((f for f in svc().files if f.name == name), None)
        if match is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(match.path, media_type="application/pdf",
                            headers={"Content-Disposition": f'inline; filename="{match.name}"'})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
