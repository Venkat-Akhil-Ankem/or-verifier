"""
evaluate.py
-----------
Run a benchmark with one or more feedback modes, store raw results as JSONL,
and compute summary metrics.

Metrics (per mode):
  pass@1         first attempt optimal on visible AND all hidden instances
  final_acc      same, after the repair loop (last attempt)
  visible_acc    final attempt optimal on the visible instance only
  repair_gain    final_acc - pass@1
  by_family / by_scale tables, failure-category counts, mean rounds
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .agent import MODES, TaskResult, run_task
from .benchmark import Task


def run_benchmark(tasks: list[Task], client, modes=("single", "repair_errors"),
                  max_rounds: int = 3, workers: int = 4, timeout_s: float = 10.0,
                  max_tokens: int = 4096, temperature=None, progress=print) -> list[TaskResult]:
    jobs = [(t, m) for m in modes for t in tasks]
    results: list[TaskResult] = []

    def work(job):
        t, m = job
        return run_task(t, client, m, max_rounds=max_rounds, timeout_s=timeout_s,
                        max_tokens=max_tokens, temperature=temperature)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            results.append(r)
            if progress:
                progress(f"[{i}/{len(jobs)}] {r.mode:14s} {r.task_id:28s} "
                         f"{'OK ' if r.final_correct else 'FAIL'} rounds={r.rounds_used}")
    return results


def save_results(results: list[TaskResult], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def load_results(path: str | Path) -> list[TaskResult]:
    from .agent import Attempt
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        d["attempts"] = [Attempt(**a) for a in d["attempts"]]
        out.append(TaskResult(**d))
    return out


def _rate(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(results: list[TaskResult]) -> dict:
    """Aggregate raw results into a nested dict: {mode: {...metrics...}}."""
    by_mode: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_mode[r.mode].append(r)

    summary = {}
    for mode in [m for m in MODES if m in by_mode]:
        rs = by_mode[mode]
        pass1 = _rate([r.first_correct for r in rs])
        final = _rate([r.final_correct for r in rs])
        fams = sorted({r.family for r in rs})
        summary[mode] = {
            "n_tasks": len(rs),
            "pass@1": pass1,
            "final_acc": final,
            "repair_gain": final - pass1,
            "visible_acc": _rate([bool(r.attempts[-1].verification and
                                       r.attempts[-1].verification["visible_ok"]) for r in rs]),
            "mean_rounds": sum(r.rounds_used for r in rs) / len(rs),
            "by_family": {f: _rate([r.final_correct for r in rs if r.family == f]) for f in fams},
            "by_scale": {("small" if s == 0 else "large"):
                         _rate([r.final_correct for r in rs if r.scale == s])
                         for s in sorted({r.scale for r in rs})},
            "failures": dict(Counter(r.final_failure or "none" for r in rs)),
        }
    return summary
