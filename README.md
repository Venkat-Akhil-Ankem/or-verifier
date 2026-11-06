# OR-Verifier

**Solver-verified evaluation and verifiable-reward RL (GRPO) for LLMs that write optimisation models.**

An LLM reads a plain-English operations-research problem and writes a Python `solve(data)` function. The code runs in a sandbox on the example instance **and on hidden instances**. An independent checker recomputes feasibility and objective value, and compares them with the **proven optimum from an exact MILP solver (HiGHS)**. The same verdict is used to

1. **evaluate** LLMs / agent strategies on a 40-task benchmark, and
2. **train** a model with **GRPO** (RL with verifiable rewards), where the solver *is* the reward function.

Eight problem families: knapsack, assignment, transportation, production planning, facility location, machine scheduling, set cover, and an **open-pit block-selection (ultimate pit)** problem from mining.

---

## What is verified and what is not (read this first)

| Component | Status |
|---|---|
| Problem generators, exact HiGHS optima, solution checkers | **Tested.** Optima cross-checked against brute-force enumeration on small instances; reference solutions pass their own checkers (`tests/test_problems.py`). |
| Sandbox (static check, timeouts, memory limit, env scrubbing), verifier, reward function, agent repair loop, metrics/report, GRPO reward adapter, advantage maths, dataset builder | **Tested** (74 pytest tests, mock LLM where needed). |
| Whole chain on the benchmark | `python scripts/selftest.py`: exact-MILP reference programs score **40/40 tasks (all visible + hidden instances)**; greedy heuristics are caught as suboptimal. |
| Calls to a real LLM (`--provider anthropic`) | **Not run by the author** (no API access when this was built). The client code is straightforward but unexercised. |
| **LLM benchmark numbers** | **None are claimed.** `results/` is empty on purpose. Your first run produces them. |
| GRPO training (`grpo/train_grpo.py`) | **Not executed** (needs a GPU). Written against the TRL API; expect to adjust arguments to your version. See `docs/grpo.md`. |

Tests ran on Python 3.13 / Linux. On Windows the per-instance alarm and resource limits are unavailable (see `docs/design.md`).

---

## Repository layout

```
or-verifier/
├── README.md
├── requirements.txt            # core: numpy, scipy, matplotlib, pytest, pulp, anthropic
├── requirements-train.txt      # GPU stack for GRPO: torch, transformers, trl, peft, ...
├── pyproject.toml              # pytest config
├── .env.example                # copy to .env and add ANTHROPIC_API_KEY
├── .gitignore
├── ci.yml                      # optional GitHub Actions workflow (see header inside)
├── src/orverifier/
│   ├── problems.py             # 8 families: generator + checker + exact HiGHS reference
│   ├── sandbox.py              # static AST check + subprocess runner with limits
│   ├── verify.py               # code -> per-instance verdict (optimal/suboptimal/infeasible/...)
│   ├── rewards.py              # verdict -> scalar RL reward (shaped or binary)
│   ├── prompts.py              # prompt construction + code extraction
│   ├── llm.py                  # Anthropic / local HF / mock / disk-cached clients
│   ├── agent.py                # generate -> verify -> repair loop (3 feedback modes)
│   ├── benchmark.py            # OR-Bench-40 builder/loader
│   ├── evaluate.py             # run benchmark, JSONL results, metrics
│   ├── report.py               # Markdown tables + chart
│   └── reference_solutions.py  # exact + heuristic solve() programs (tests, offline demo)
├── benchmark/or_bench_v1.json  # 40 tasks with exact optima (generated, committed)
├── scripts/
│   ├── build_benchmark.py      # regenerate the benchmark
│   ├── selftest.py             # offline end-to-end check, no LLM needed
│   ├── run_eval.py             # evaluate a model (anthropic | hf | mock)
│   └── make_report.py          # rebuild report from saved results
├── grpo/
│   ├── advantages.py           # GRPO maths in NumPy (readable reference)
│   ├── dataset.py              # training prompts (seeds disjoint from benchmark)
│   ├── reward_fn.py            # TRL-compatible reward function
│   └── train_grpo.py           # TRL + LoRA training script  (NOT executed)
├── tests/                      # 74 tests
├── docs/
│   ├── design.md               # architecture, reward design, sandbox limits, limitations
│   └── grpo.md                 # GRPO explained, how to run, before/after protocol
└── results/                    # your evaluation outputs go here
```

---

## Quick start (laptop, no GPU)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # 74 tests (~20 s)
python scripts/selftest.py           # reference solutions verified on all 40 tasks
```

### Evaluate Claude

```bash
cp .env.example .env                 # then edit .env and set ANTHROPIC_API_KEY
python scripts/run_eval.py --provider anthropic --families knapsack set_cover --limit 1   # cheap smoke test
python scripts/run_eval.py --provider anthropic --model claude-sonnet-5-5                 # full run, 3 modes
```

Outputs `results/<run>.jsonl` (every attempt, code and verdict), `.md` (tables) and `.png` (chart by family). Responses are cached in `.cache/`, so re-running is free. The model name is configurable (`--model` or `OR_MODEL`); use whichever current model you have access to.

### Offline demo of the pipeline

`python scripts/run_eval.py --provider mock` replays a scripted "perfect model". It exists only to demonstrate the plumbing. Its 100% is meaningless as a result; don't publish it.

---

## How it works

```
Natural-language problem + 1 example instance
        │
        ▼
      LLM  ──►  ```python def solve(data): ... ```
        │
        ▼
 static check (imports allow-list, no eval/open/dunders)
        │
        ▼
 sandbox subprocess: run solve() on [visible, hidden₁, hidden₂, hidden₃]
        │   (timeouts, memory/CPU limits, no API keys in env)
        ▼
 checker: recompute feasibility + objective from the returned solution
        │   compare with proven optimum (HiGHS)
        ▼
 verdict per instance: optimal │ suboptimal │ infeasible │ malformed │ error │ timeout │ rejected
        │
        ├──► repair loop: send crash/infeasibility info from the VISIBLE instance back to the LLM
        ├──► metrics + report
        └──► reward ∈ [0,1] ──► GRPO
```

### Three feedback modes (the agent ablation)

| Mode | The model is told |
|---|---|
| `single` | nothing; one attempt |
| `repair_errors` | crashes, wrong output format, constraint violations on the example (realistic) |
| `repair_oracle` | additionally "feasible but not optimal; optimum = X" (upper bound) |

Hidden-instance results are **never** shown to the model, so repair cannot leak the test set.

### Metrics

`pass@1` (first attempt optimal on all instances), `final accuracy` after repair, `repair gain`, accuracy by family and by size (small vs large), and a **failure taxonomy** of the final attempt: `static_check`, `runtime_error`, `timeout`, `malformed_output`, `infeasible`, `suboptimal`, `fails_hidden` (right on the example, wrong elsewhere), `no_code`.

### GRPO in short

Sample G completions per prompt → score each with the solver-based reward → advantage = reward standardised within the group → clipped policy-gradient update (no critic, no reward model). Details, run instructions and the before/after protocol are in [`docs/grpo.md`](docs/grpo.md).

---

## Honest limitations

* One wording per problem family; tasks differ in data and size, not phrasing.
* 40 tasks → small differences are noise.
* Textbook-scale problems; the mining problem is a toy ultimate-pit model, not MineLib scale.
* The Python sandbox is a guard rail, not a security boundary; use a container for untrusted code.
* GRPO training is untested code; no improvement is claimed.

More in [`docs/design.md`](docs/design.md).

## Ideas for extension

Held-out-family generalisation test for GRPO; more phrasings per problem; MineLib-derived pit instances; comparison with NL4Opt / OptiMUS-style benchmarks; error-category-targeted prompting.

## License

MIT (add a `LICENSE` file when you publish).
