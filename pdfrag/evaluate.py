"""Run the sample evaluation and print a markdown report.

    python -m pdfrag.evaluate                 # print the report
    python -m pdfrag.evaluate --write         # also write docs/eval-results.md
    python -m pdfrag.evaluate --update-readme # also refresh the table in README.md
    python -m pdfrag.evaluate --sweep         # answered vs refused for several thresholds

Uses the same settings as the app (LLM_PROVIDER, DOCS_DIR, MIN_CONFIDENCE...),
so you can compare extractive mode with an LLM provider on the same questions.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import yaml

from .answer import RagPipeline
from .config import ROOT, Settings
from .service import Service, build_service

QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS = ROOT / "docs" / "eval-results.md"
README = ROOT / "README.md"
START, END = "<!-- eval:start -->", "<!-- eval:end -->"


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def run(settings: Settings, questions_file: Path = QUESTIONS, service: Service | None = None) -> dict:
    service = service or build_service(settings)
    pipeline = service.pipeline
    items = load_questions(questions_file)
    rows = []
    for item in items:
        res = pipeline.ask(item["question"])
        unanswerable = bool(item.get("unanswerable"))
        expected = {(e["source"], e["page"]) for e in item.get("expected", [])}
        retrieved = {(s.source, s.page) for s in res.snippets}
        cited = {(c.source, c.page) for c in res.citations}
        must = item.get("must_contain", "")
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "unanswerable": unanswerable,
            "refused": res.refused,
            "reason": res.reason or "",
            "confidence": res.confidence,
            "retrieval_hit": bool(expected & retrieved),
            "citation_ok": (not res.refused) and bool(expected & cited),
            "fact_ok": (not res.refused) and must.lower() in res.answer.lower(),
            "cited": sorted(cited),
            "answer": res.answer,
        })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    answered = [r for r in ans if not r["refused"]]
    summary = {
        "mode": pipeline.mode,
        "questions": len(rows),
        "retrieval_hit": _pct(sum(r["retrieval_hit"] for r in ans), len(ans)),
        "answered": _pct(len(answered), len(ans)),
        "citation_ok": _pct(sum(r["citation_ok"] for r in answered), len(answered)),
        "fact_ok": _pct(sum(r["fact_ok"] for r in answered), len(answered)),
        "correct_refusal": _pct(sum(r["refused"] for r in una), len(una)),
        "wrong_answers_unanswerable": sum(not r["refused"] for r in una),
    }
    return {"summary": summary, "rows": rows, "settings": settings}


def summary_table(result: dict) -> str:
    s = result["summary"]
    st = result["settings"]
    return "\n".join([
        f"Mode: **{s['mode']}** · top_k={st.top_k} · MIN_CONFIDENCE={st.min_confidence} · "
        f"{s['questions']} questions in `eval/questions.yaml`",
        "",
        "| Metric | What it measures | Result |",
        "|---|---|---|",
        f"| Retrieval hit rate | Answerable questions where the expected page is in the top-{st.top_k} snippets | {s['retrieval_hit']} |",
        f"| Answered | Answerable questions the bot answered instead of refusing | {s['answered']} |",
        f"| Citation correctness | Answers that cite the expected document and page | {s['citation_ok']} |",
        f"| Key fact present | Answers that contain the expected fact (e.g. \"30 days\") | {s['fact_ok']} |",
        f"| Correct refusal rate | Unanswerable questions where the bot said it could not find the answer | {s['correct_refusal']} |",
    ])


def detail_table(result: dict) -> str:
    lines = ["| ID | Question | Expected | Result | Confidence | Cited |", "|---|---|---|---|---|---|"]
    for r in result["rows"]:
        expected = "refuse" if r["unanswerable"] else "answer"
        if r["refused"]:
            got = f"refused ({r['reason']})"
        else:
            got = "answered" + ("" if r["unanswerable"] else (" ✓" if r["citation_ok"] and r["fact_ok"] else " ✗"))
        cited = ", ".join(f"{s} p.{p}" for s, p in r["cited"]) or "-"
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {r['id']} | {q} | {expected} | {got} | {r['confidence']:.2f} | {cited} |")
    return "\n".join(lines)


def sweep(settings: Settings) -> str:
    """Show how MIN_CONFIDENCE trades answered questions against correct refusals."""
    service = build_service(settings)
    base = service.pipeline
    lines = ["| MIN_CONFIDENCE | Answered (answerable) | Correct + cited | Correct refusals (unanswerable) |",
             "|---|---|---|---|"]
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        st = dataclasses.replace(settings, min_confidence=t)
        service.pipeline = RagPipeline(base.index, st, base.llm, base.mode)
        s = run(st, service=service)
        rows = s["rows"]
        ans = [r for r in rows if not r["unanswerable"]]
        good = sum(r["citation_ok"] and r["fact_ok"] for r in ans)
        lines.append(f"| {t:.2f} | {s['summary']['answered']} | {_pct(good, len(ans))} | "
                     f"{s['summary']['correct_refusal']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write docs/eval-results.md")
    parser.add_argument("--update-readme", action="store_true", help="refresh the summary table in README.md")
    parser.add_argument("--sweep", action="store_true", help="compare several MIN_CONFIDENCE values")
    args = parser.parse_args()

    if args.sweep:
        print(sweep(Settings.from_env()))
        return

    result = run(Settings.from_env())
    summary = summary_table(result)
    report = "# Evaluation results\n\nGenerated by `python -m pdfrag.evaluate --write`.\n\n" \
             + summary + "\n\n## Per question\n\n" + detail_table(result) + "\n"
    print(report)

    if args.write or args.update_readme:
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(report, encoding="utf-8", newline="\n")
    if args.update_readme and README.exists():
        text = README.read_text(encoding="utf-8")
        block = f"{START}\n{summary}\n{END}"
        text, n = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        if n:
            README.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
