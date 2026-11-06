"""Rebuild the Markdown report + chart from a saved results JSONL (no LLM calls).

Usage:  python scripts/make_report.py results/<run>.jsonl
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.evaluate import load_results, summarize  # noqa: E402
from orverifier.report import plot_by_family, to_markdown  # noqa: E402

if __name__ == "__main__":
    path = Path(sys.argv[1])
    summary = summarize(load_results(path))
    path.with_suffix(".md").write_text(to_markdown(summary, f"Results: {path.stem}"))
    plot_by_family(summary, path.with_suffix(".png"))
    print(f"Wrote {path.with_suffix('.md')} and {path.with_suffix('.png')}")
