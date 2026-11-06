"""
Offline self-test of the whole verification chain (no LLM, no API key, no GPU).

  1. Every reference program must score 100% on every benchmark task
     (visible + hidden). This validates generators, sandbox, checkers and optima together.
  2. Heuristic programs (greedy etc.) must be caught as wrong on at least some tasks.

Usage:  python scripts/selftest.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.benchmark import load_benchmark  # noqa: E402
from orverifier.reference_solutions import CORRECT, HEURISTIC  # noqa: E402
from orverifier.verify import verify_code  # noqa: E402


def main() -> int:
    tasks = load_benchmark(ROOT / "benchmark" / "or_bench_v1.json")
    print(f"Loaded {len(tasks)} tasks\n")

    bad = 0
    print("Reference (exact MILP) programs - expected 100% on all instances:")
    for fam in CORRECT:
        ts = [t for t in tasks if t.family == fam]
        ok = 0
        for t in ts:
            v = verify_code(CORRECT[fam], fam, t.instances)
            ok += v.all_correct
            if not v.all_correct:
                print("   FAIL", t.task_id, [o.status + ":" + o.message for o in v.outcomes])
        print(f"  {fam:22s} {ok}/{len(ts)}")
        bad += len(ts) - ok

    print("\nHeuristic programs - expected to be caught as suboptimal on some tasks:")
    for fam, code in HEURISTIC.items():
        ts = [t for t in tasks if t.family == fam]
        cats = Counter()
        for t in ts:
            v = verify_code(code, fam, t.instances)
            cats[v.failure_category() or "correct"] += 1
        print(f"  {fam:22s} {dict(cats)}")

    print("\nSELFTEST", "PASSED" if bad == 0 else f"FAILED ({bad} reference failures)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
