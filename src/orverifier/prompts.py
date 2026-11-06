"""
prompts.py
----------
Prompt construction and code extraction. Shared by the evaluation harness and
the GRPO training data so that train/eval prompts have identical format.
"""

from __future__ import annotations

import json
import re

from .problems import FAMILIES, Instance

SYSTEM_PROMPT = (
    "You are an expert in operations research and mathematical optimisation. "
    "You write correct, efficient Python and you formulate problems exactly."
)

_USER_TEMPLATE = """Problem
-------
{statement}

Input
-----
Your function receives one argument `data` (a dict) with:
{data_spec}

Output
------
Return {solution_spec}.

Rules
-----
- Define a top-level function `solve(data)`; do not read files or the network.
- You may use: numpy, scipy (e.g. scipy.optimize.milp / linprog, which use the HiGHS solver),
  pulp, and the Python standard library (math, itertools, collections, heapq, ...).
- The returned solution must be OPTIMAL, not just feasible. It will be checked by an
  independent verifier against the proven optimum.
- Your function will also be run on OTHER hidden instances with the same format (same
  problem, different numbers, possibly larger), with a time limit of 10 seconds each.
  Do not hard-code the example.

Example instance (data)
-----------------------
{example}

Reply with ONE ```python code block containing the complete solution and nothing else."""


def build_user_prompt(family: str, visible: Instance) -> str:
    fam = FAMILIES[family]
    return _USER_TEMPLATE.format(
        statement=fam.statement,
        data_spec=fam.data_spec,
        solution_spec=fam.solution_spec,
        example=json.dumps(visible.data),
    )


def build_messages(family: str, visible: Instance) -> list[dict]:
    """Conversational prompt (system + user), used for GRPO training data."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(family, visible)},
    ]


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Return the code from the first fenced block that defines `solve`, else None."""
    blocks = _FENCE.findall(text or "")
    for b in blocks:
        if "def solve" in b:
            return b.strip("\n")
    if blocks:
        return blocks[0].strip("\n")
    if "def solve" in (text or ""):           # unfenced fallback
        return text.strip()
    return None
