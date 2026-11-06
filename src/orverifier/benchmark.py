"""
benchmark.py
------------
OR-Bench-40: 8 problem families x 5 tasks (3 small + 2 large) = 40 tasks.
Each task = one *visible* instance (shown in the prompt) + hidden instances of
the same size scale (used only for scoring).

Benchmark seeds are < TRAIN_SEED_START; GRPO training seeds start at
TRAIN_SEED_START so train and test instances never overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .problems import FAMILIES, Instance, make_instance

TRAIN_SEED_START = 10_000
N_SMALL, N_LARGE, N_HIDDEN = 3, 2, 3


@dataclass
class Task:
    task_id: str
    family: str
    scale: int
    visible: Instance
    hidden: list[Instance]

    @property
    def instances(self) -> list[Instance]:
        return [self.visible, *self.hidden]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "family": self.family, "scale": self.scale,
                "visible": self.visible.to_dict(), "hidden": [h.to_dict() for h in self.hidden]}

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(d["task_id"], d["family"], d["scale"], Instance.from_dict(d["visible"]),
                    [Instance.from_dict(h) for h in d["hidden"]])


def build_benchmark(families: list[str] | None = None, n_small: int = N_SMALL,
                    n_large: int = N_LARGE, n_hidden: int = N_HIDDEN) -> list[Task]:
    tasks: list[Task] = []
    for fam in families or list(FAMILIES):
        k = 0
        for scale, count in ((0, n_small), (1, n_large)):
            for i in range(count):
                base = 100 * (k + 1)                       # visible seed = base, hidden = base+1..
                assert base + n_hidden < TRAIN_SEED_START
                visible = make_instance(fam, base, scale)
                hidden = [make_instance(fam, base + 1 + h, scale) for h in range(n_hidden)]
                tasks.append(Task(f"{fam}-{'S' if scale == 0 else 'L'}{i + 1}", fam, scale,
                                  visible, hidden))
                k += 1
    return tasks


def save_benchmark(tasks: list[Task], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([t.to_dict() for t in tasks]))


def load_benchmark(path: str | Path) -> list[Task]:
    return [Task.from_dict(d) for d in json.loads(Path(path).read_text())]
