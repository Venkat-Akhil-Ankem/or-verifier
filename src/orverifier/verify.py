"""
verify.py
---------
Turn "some Python code" into a verdict:

    code  --static check-->  --sandbox run on visible + hidden instances-->
          --independent feasibility/objective check-->  --compare with proven optimum-->  Verification

The first instance in the list is the *visible* one (shown to the model in its
prompt); the rest are *hidden* (never shown, never used in repair feedback) and
guard against hard-coded or over-fitted answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .problems import FAMILIES, OPT_TOL, Instance
from .sandbox import run_solve, static_check

# Outcome statuses
OPTIMAL = "optimal"          # feasible and matches the proven optimum
SUBOPTIMAL = "suboptimal"    # feasible but worse than the optimum
INFEASIBLE = "infeasible"    # well-formed but violates a constraint
MALFORMED = "malformed"      # wrong return type/shape
ERROR = "error"              # exception while running
TIMEOUT = "timeout"
REJECTED = "rejected"        # static check failed - never executed

RUN_FAILURES = {ERROR, TIMEOUT, REJECTED, MALFORMED}


@dataclass
class InstanceOutcome:
    status: str
    objective: float | None = None
    optimum: float | None = None
    rel_gap: float | None = None      # >0 means worse than optimum
    message: str = ""


@dataclass
class Verification:
    family: str
    static_errors: list[str] = field(default_factory=list)
    outcomes: list[InstanceOutcome] = field(default_factory=list)   # [0] visible, rest hidden

    # --- convenience views -------------------------------------------------
    @property
    def visible(self) -> InstanceOutcome:
        return self.outcomes[0]

    @property
    def hidden(self) -> list[InstanceOutcome]:
        return self.outcomes[1:]

    @property
    def visible_ok(self) -> bool:
        return self.visible.status == OPTIMAL

    @property
    def hidden_ok_frac(self) -> float:
        h = self.hidden
        return sum(o.status == OPTIMAL for o in h) / len(h) if h else 1.0

    @property
    def all_correct(self) -> bool:
        return bool(self.outcomes) and all(o.status == OPTIMAL for o in self.outcomes)

    @property
    def runs_on_visible(self) -> bool:
        return self.visible.status not in RUN_FAILURES

    @property
    def feasible_on_visible(self) -> bool:
        return self.visible.status in (OPTIMAL, SUBOPTIMAL)

    # --- diagnostics -------------------------------------------------------
    def failure_category(self) -> str | None:
        """Why did this attempt fail? None if it is fully correct."""
        if self.all_correct:
            return None
        if self.static_errors:
            return "static_check"
        v = self.visible.status
        if v == ERROR:
            return "runtime_error"
        if v == TIMEOUT:
            return "timeout"
        if v == MALFORMED:
            return "malformed_output"
        if v == INFEASIBLE:
            return "infeasible"
        if v == SUBOPTIMAL:
            return "suboptimal"
        return "fails_hidden"       # right on the shown instance, wrong elsewhere

    def feedback(self, reveal_optimum: bool = False) -> str | None:
        """
        Message for the repair loop, built from the VISIBLE instance only.
        Returns None when there is nothing actionable to report.
        """
        if self.static_errors:
            return "Your code was rejected before running:\n- " + "\n- ".join(self.static_errors)
        v = self.visible
        if v.status == ERROR:
            return f"Your code crashed on the example instance: {v.message}"
        if v.status == TIMEOUT:
            return "Your code exceeded the time limit on the example instance. Use a more efficient method."
        if v.status == MALFORMED:
            return f"Your return value has the wrong format: {v.message}"
        if v.status == INFEASIBLE:
            return f"Your solution is infeasible on the example instance: {v.message}"
        if v.status == SUBOPTIMAL and reveal_optimum:
            return (f"Your solution is feasible with objective {v.objective:.6g}, but the best "
                    f"possible objective is {v.optimum:.6g}. The model or algorithm is not optimal.")
        return None

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "static_errors": self.static_errors,
            "outcomes": [asdict(o) for o in self.outcomes],
            "all_correct": self.all_correct,
            "visible_ok": self.visible_ok if self.outcomes else False,
            "hidden_ok_frac": self.hidden_ok_frac if self.outcomes else 0.0,
            "failure_category": self.failure_category(),
        }


# ──────────────────────────────────────────────────────────────────────────────

def judge(family: str, instance: Instance, solution) -> InstanceOutcome:
    """Check one returned solution against one instance."""
    fam = FAMILIES[family]
    try:
        feasible, obj, msg = fam.evaluate(instance.data, solution)
    except Exception as e:                     # malformed return value
        return InstanceOutcome(MALFORMED, message=str(e)[:300], optimum=instance.optimum)

    if not feasible:
        return InstanceOutcome(INFEASIBLE, message=msg, optimum=instance.optimum)

    opt = instance.optimum
    denom = max(1.0, abs(opt))
    gap = (obj - opt) / denom if fam.sense == "min" else (opt - obj) / denom
    if not math.isfinite(gap):
        return InstanceOutcome(MALFORMED, message="objective is not finite", optimum=opt)
    status = OPTIMAL if gap <= OPT_TOL else SUBOPTIMAL
    note = "better than reference optimum - check reference/checker" if gap < -1e-4 else "ok"
    return InstanceOutcome(status, objective=obj, optimum=opt, rel_gap=gap, message=note)


def verify_code(code: str, family: str, instances: list[Instance],
                timeout_s: float = 10.0, mem_mb: int = 3072) -> Verification:
    """Verify `code` on [visible, *hidden] instances."""
    ver = Verification(family=family)

    errors = static_check(code)
    if errors:
        ver.static_errors = errors
        ver.outcomes = [InstanceOutcome(REJECTED, optimum=i.optimum, message="; ".join(errors))
                        for i in instances]
        return ver

    raw = run_solve(code, [i.data for i in instances], timeout_s=timeout_s, mem_mb=mem_mb)
    for inst, r in zip(instances, raw):
        if r["ok"]:
            ver.outcomes.append(judge(family, inst, r["solution"]))
        elif r["error"] == "timeout":
            ver.outcomes.append(InstanceOutcome(TIMEOUT, optimum=inst.optimum,
                                                message=f"exceeded {timeout_s}s"))
        else:
            ver.outcomes.append(InstanceOutcome(ERROR, optimum=inst.optimum, message=r["error"]))
    return ver
