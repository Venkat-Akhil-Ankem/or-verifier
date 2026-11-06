"""
reward_fn.py
------------
Reward function in the signature TRL's GRPOTrainer expects:

    reward_func(prompts, completions, **dataset_columns) -> list[float]

Dataset columns we use (passed by TRL as keyword lists): `family`, `instances`.
Completions may be plain strings or conversational ([{"role":..,"content":..}]).

Each completion is executed in the sandbox (1 subprocess, visible + hidden
instances), so scoring is parallelised with threads.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.problems import Instance  # noqa: E402
from orverifier.rewards import score_completion  # noqa: E402


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:          # conversational format
        last = completion[-1]
        return last["content"] if isinstance(last, dict) else str(last)
    return str(completion)


def make_reward_fn(timeout_s: float = 10.0, workers: int = 8, binary: bool = False):
    def or_reward(prompts=None, completions=None, family=None, instances=None, **kwargs):
        jobs = []
        for comp, fam, inst_json in zip(completions, family, instances):
            insts = [Instance.from_dict(d) for d in json.loads(inst_json)]
            jobs.append((_text(comp), fam, insts))

        def score(job):
            text, fam, insts = job
            try:
                return score_completion(text, fam, insts, timeout_s=timeout_s, binary=binary)
            except Exception:                                   # never crash training
                return 0.0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(score, jobs))

    or_reward.__name__ = "or_reward_binary" if binary else "or_reward"
    return or_reward
