"""
agent.py
--------
The text-to-optimisation agent loop:

    prompt -> LLM -> extract code -> sandbox + verify -> (feedback) -> LLM -> ...

Feedback modes (an ablation on how much the model is told):

  single          one shot, no repair                          (baseline)
  repair_errors   told about crashes / bad format / infeasibility on the VISIBLE
                  instance only. Realistic: no optimum is needed.
  repair_oracle   additionally told when a feasible answer is not optimal and what
                  the optimum is. Upper bound - needs a known optimum.

Hidden instances are verified for scoring every attempt but their results are
NEVER shown to the model (no test-set leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .benchmark import Task
from .prompts import SYSTEM_PROMPT, build_user_prompt, extract_code
from .verify import Verification, verify_code

MODES = ("single", "repair_errors", "repair_oracle")


@dataclass
class Attempt:
    response: str
    code: str | None
    verification: dict | None          # Verification.to_dict(); None if no code was found
    all_correct: bool
    failure: str | None
    feedback: str | None = None        # message sent to the model after this attempt


@dataclass
class TaskResult:
    task_id: str
    family: str
    scale: int
    mode: str
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def first_correct(self) -> bool:
        return self.attempts[0].all_correct

    @property
    def final_correct(self) -> bool:
        return self.attempts[-1].all_correct

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    @property
    def final_failure(self) -> str | None:
        return self.attempts[-1].failure


def run_task(task: Task, client, mode: str = "repair_errors", max_rounds: int = 3,
             timeout_s: float = 10.0, max_tokens: int = 4096, temperature=None) -> TaskResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rounds = 1 if mode == "single" else max_rounds
    reveal = mode == "repair_oracle"

    messages = [{"role": "user", "content": build_user_prompt(task.family, task.visible)}]
    result = TaskResult(task.task_id, task.family, task.scale, mode)

    for r in range(rounds):
        text = client.complete(messages, SYSTEM_PROMPT, max_tokens=max_tokens, temperature=temperature)
        code = extract_code(text)

        ver: Verification | None = None
        if code is None:
            feedback = ("I could not find a ```python code block defining `solve(data)` in your "
                        "reply. Reply with exactly one python code block.")
            attempt = Attempt(text, None, None, False, "no_code", feedback)
        else:
            ver = verify_code(code, task.family, task.instances, timeout_s=timeout_s)
            feedback = ver.feedback(reveal_optimum=reveal)       # visible-only information
            attempt = Attempt(text, code, ver.to_dict(), ver.all_correct,
                              ver.failure_category(), feedback)
        result.attempts.append(attempt)

        # Stop when the model has nothing actionable left to fix.
        if feedback is None or r == rounds - 1:
            attempt.feedback = None          # nothing was sent back to the model
            break
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": feedback +
                         "\n\nFix the problem and reply with the complete corrected code in one ```python block."})
    return result
