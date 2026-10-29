"""Run the web app: python -m pdfrag  (then open http://localhost:8000)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("pdfrag.api:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
