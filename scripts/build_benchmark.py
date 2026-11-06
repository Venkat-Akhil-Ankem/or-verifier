"""Generate benchmark/or_bench_v1.json (40 tasks, exact optima from HiGHS).

Usage:  python scripts/build_benchmark.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.benchmark import build_benchmark, save_benchmark  # noqa: E402

if __name__ == "__main__":
    tasks = build_benchmark()
    out = ROOT / "benchmark" / "or_bench_v1.json"
    save_benchmark(tasks, out)
    fams = sorted({t.family for t in tasks})
    print(f"Wrote {len(tasks)} tasks ({len(fams)} families) -> {out}")
