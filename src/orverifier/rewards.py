"""
rewards.py
----------
Verifiable reward for RL (GRPO / RLVR) built directly on `verify_code`.

Shaped reward (max 1.0):

    0.10  response contains code that passes the static check
    0.10  code runs on the visible instance without crash/timeout/bad format
    0.20  visible solution is feasible
    0.30  visible solution is optimal
    0.30  * fraction of hidden instances solved optimally   (anti-memorisation term)

The reward equals 1.0 if and only if the code is optimal on every instance.
`reward_binary` is the plain 0/1 alternative (useful as an ablation).
"""

from __future__ import annotations

from .prompts import extract_code
from .problems import Instance
from .verify import Verification, verify_code

WEIGHTS = {"format": 0.10, "runs": 0.10, "feasible": 0.20, "visible_opt": 0.30, "hidden": 0.30}


def reward_components(ver: Verification | None) -> dict[str, float]:
    """Break the shaped reward into its terms (None = no code extracted)."""
    comp = {k: 0.0 for k in WEIGHTS}
    if ver is None or ver.static_errors:
        return comp
    comp["format"] = WEIGHTS["format"]
    if ver.runs_on_visible:
        comp["runs"] = WEIGHTS["runs"]
    if ver.feasible_on_visible:
        comp["feasible"] = WEIGHTS["feasible"]
    if ver.visible_ok:
        comp["visible_opt"] = WEIGHTS["visible_opt"]
    comp["hidden"] = WEIGHTS["hidden"] * ver.hidden_ok_frac if ver.visible_ok else 0.0
    return comp


def reward_shaped(ver: Verification | None) -> float:
    return round(sum(reward_components(ver).values()), 10)


def reward_binary(ver: Verification | None) -> float:
    return 1.0 if (ver is not None and ver.all_correct) else 0.0


def score_completion(text: str, family: str, instances: list[Instance],
                     timeout_s: float = 10.0, binary: bool = False) -> float:
    """Full pipeline: raw model text -> reward."""
    code = extract_code(text)
    ver = verify_code(code, family, instances, timeout_s=timeout_s) if code else None
    return reward_binary(ver) if binary else reward_shaped(ver)
