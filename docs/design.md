# Design notes

## The core idea: a solver is a perfect judge

In most LLM evaluation the "grader" is itself approximate (another LLM, string
matching, human labels). In optimisation it does not have to be:

* every problem family has an exact solver (SciPy/HiGHS MILP) that yields the
  **proven optimum**;
* every family has a **checker** that recomputes feasibility and the objective
  *from the returned solution*. The model's own claims are never trusted.

So `verdict = f(returned solution, instance)` is exact, cheap and automatic. That
single function is used twice:

| Use | How |
|---|---|
| **Evaluation** | benchmark accuracy, repair-loop ablations, failure analysis |
| **Reinforcement learning** | `rewards.py` turns the verdict into a scalar reward for GRPO (RL with Verifiable Rewards) |

## Data flow

```
 problems.py ──generate──► Instance(data, optimum)           (optimum from HiGHS, cached)
      │
 prompts.py ──► prompt = statement + data spec + 1 visible example instance
      │
 LLM (llm.py) ──► reply ──► extract_code
      │
 sandbox.py:  static AST check ─► subprocess (rlimits, scrubbed env, per-instance timeout)
      │                          runs solve(data) on [visible, hidden_1..hidden_k]
      ▼
 verify.py:  checker.evaluate(instance, solution) ─► status ∈ {optimal, suboptimal,
             infeasible, malformed, error, timeout, rejected} + relative gap to optimum
      │
      ├─► agent.py     visible-only feedback ─► repair loop (3 modes)
      ├─► evaluate.py  metrics ─► report.py (Markdown + chart)
      └─► rewards.py   scalar reward ─► grpo/reward_fn.py ─► TRL GRPOTrainer
```

## Why hidden instances?

Each prompt shows **one** instance. If only that instance were checked, a model
could return a hard-coded answer. Each task therefore also holds hidden instances
of the same family and size. A solution must be optimal on *all* of them, which
forces a general formulation. Hidden results are used for scoring and for the RL
reward, but are **never** put into repair feedback.

Large tasks (e.g. 40-item knapsack, 20x20 assignment, 14-job scheduling) are
meant to break brute-force enumeration: a 10-second per-instance limit applies.

## Feedback modes (agent ablation)

| Mode | Model is told | Realistic? |
|---|---|---|
| `single` | nothing, one attempt | baseline |
| `repair_errors` | crashes, bad output format, constraint violations (visible instance) | yes - a user would see these |
| `repair_oracle` | additionally "feasible but not optimal; optimum is X" | upper bound, needs a known optimum |

The loop stops as soon as the visible instance gives nothing actionable.

## Reward design (GRPO)

| Term | Weight | Condition |
|---|---|---|
| format | 0.10 | code extracted and passes static check |
| runs | 0.10 | executes on visible instance, valid output |
| feasible | 0.20 | visible solution feasible |
| visible optimal | 0.30 | matches proven optimum |
| hidden | 0.30 x fraction | hidden instances solved optimally (only if visible optimal) |

Dense partial credit helps early training when a small model rarely gets
everything right; reward = 1.0 only for fully correct code. `--binary-reward`
in `train_grpo.py` gives a 0/1 alternative for ablation. Rewards in the
partial-credit terms can be gamed in principle (e.g. emitting trivially
feasible code earns 0.4) - this is why the optimality terms carry most weight
and why you should always inspect sampled completions during training.

## Sandbox: what it is and is not

Protects against: infinite loops, memory blow-ups, crashes, stray prints, mutation
of shared data, accidental file/network access (imports blocked), API keys in the
environment (scrubbed).

Does **not** protect against a determined attacker: Python-level sandboxes can be
escaped (e.g. through allowed libraries such as numpy/scipy/pulp, which can reach
the OS). If you evaluate untrusted or RL-trained models at scale, run the whole
harness in a container with no network (`docker run --network none ...`) and a
read-only filesystem. On Windows the per-instance alarm and resource limits are
unavailable; only the overall hard timeout applies - use WSL or Docker.

## Known limitations (be upfront about these in interviews)

* 8 problem families with **one wording each**; variation between tasks is in the
  data and size, not in the phrasing. This measures formulation skill on textbook
  structures, not robustness to messy real-world descriptions.
* 40 tasks is small: differences of a few points between models/modes are noise.
  Use `--families`/seeds or enlarge `N_SMALL/N_LARGE` in `benchmark.py` for tighter estimates.
* Problems are textbook-size. Real mine-scheduling models (MineLib-scale, millions
  of variables) are out of scope; pit block selection here is a toy ultimate-pit LP.
* Temperature is left at the API default unless `--temperature` is given, so API
  results are not bit-reproducible; the on-disk cache (`.cache/`) makes *re-scoring*
  reproducible.
