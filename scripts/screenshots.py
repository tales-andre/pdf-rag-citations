"""Capture the README screenshots from the running app.

Starts the app in extractive mode, opens it in an installed Microsoft Edge or
Google Chrome through Playwright (no browser download), and saves PNGs to docs/.

Run: python scripts/screenshots.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"

SHOTS = [
    ("screenshot-answer.png", "How many days do customers have to return an unopened item?"),
    ("screenshot-refusal.png", "What is the warranty on the X500 drone?"),
]


def wait_for_server(timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=1)
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("server did not start")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    env = dict(os.environ, LLM_PROVIDER="none")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "pdfrag.api:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome"):
                try:
                    browser = p.chromium.launch(channel=channel, headless=True)
                    break
                except Exception:
                    continue
            if browser is None:
                raise RuntimeError("Install Microsoft Edge or Google Chrome to take screenshots.")
            page = browser.new_page(viewport={"width": 1360, "height": 860}, device_scale_factor=1)
            for name, question in SHOTS:
                page.goto(f"{BASE}/?q={quote(question)}")
                page.wait_for_selector(".status")
                page.wait_for_function("document.getElementById('mode').textContent !== 'loading…'")
                page.screenshot(path=str(DOCS / name), full_page=True)
                print("saved", DOCS / name)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
