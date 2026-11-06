"""
dataset.py
----------
Build GRPO training prompts. Each record holds the chat prompt (system + user) and
the instances (visible first, then hidden, with exact optima) the reward function
needs to score completions.

Train seeds start at TRAIN_SEED_START so they can never coincide with benchmark
seeds; we additionally drop any instance whose data equals a benchmark instance.

Usage:  python grpo/dataset.py --per-family 150 --out grpo/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.benchmark import TRAIN_SEED_START, load_benchmark  # noqa: E402
from orverifier.problems import FAMILIES, make_instance  # noqa: E402
from orverifier.prompts import build_messages  # noqa: E402


def _key(data: dict) -> str:
    return json.dumps(data, sort_keys=True)


def build_records(per_family: int = 100, n_hidden: int = 3, large_fraction: float = 0.3,
                  seed: int = 0, bench_path: str | Path | None = None) -> list[dict]:
    bench_path = bench_path or ROOT / "benchmark" / "or_bench_v1.json"
    banned = set()
    if Path(bench_path).exists():
        for t in load_benchmark(bench_path):
            banned.update(_key(i.data) for i in t.instances)

    rng = random.Random(seed)
    records, next_seed = [], TRAIN_SEED_START
    for fam in FAMILIES:
        made = 0
        while made < per_family:
            scale = 1 if rng.random() < large_fraction else 0
            insts = [make_instance(fam, next_seed + k, scale) for k in range(1 + n_hidden)]
            next_seed += 1 + n_hidden
            if any(_key(i.data) in banned for i in insts):
                continue
            records.append({
                "prompt": build_messages(fam, insts[0]),
                "family": fam,
                "instances": json.dumps([i.to_dict() for i in insts]),
            })
            made += 1
    rng.shuffle(records)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "grpo" / "train.jsonl"))
    args = ap.parse_args()
    recs = build_records(args.per_family)
    with open(args.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(recs)} training prompts -> {args.out}")


if __name__ == "__main__":
    main()
