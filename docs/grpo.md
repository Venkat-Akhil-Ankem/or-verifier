# GRPO with solver-verified rewards (RLVR)

> **Status: not run by the author.** The reward function, dataset builder and
> advantage maths are unit-tested (`tests/test_grpo.py`). `grpo/train_grpo.py`
> (TRL + LoRA) was written against the TRL API but has **never been executed**
> because no GPU was available. Treat the first run as debugging. I make no claim
> about how much training improves a model - measure it with the before/after
> protocol below.

## What GRPO is, in one paragraph

For each prompt the policy samples a **group** of G completions. Each completion is
scored by the reward function. The advantage of a completion is its reward
standardised within its group: `A_i = (r_i - mean) / (std + eps)`. The policy is
updated with a PPO-style clipped objective weighted by these advantages, usually
with a KL penalty to a reference model. There is **no value network** (critic),
which makes it cheaper than PPO, and with a programmatic reward there is **no
reward model** either - the solver is the reward. See `grpo/advantages.py` for the
formulas in 20 lines of NumPy.

## Why this task suits it

* The reward is exact and automatic (proven optimum + independent checker).
* Different samples of the same prompt typically differ in quality (crash /
  infeasible / suboptimal / optimal), so groups have non-zero variance - the signal
  GRPO needs.
* The hidden-instance term pushes the policy toward *general* formulations
  rather than memorised answers.

## How to run (GPU needed)

```bash
pip install -r requirements-train.txt
python grpo/dataset.py --per-family 150          # 1,200 prompts -> grpo/train.jsonl
python grpo/train_grpo.py --max-steps 50         # smoke test first!
python grpo/train_grpo.py --max-steps 300        # real run
```

Reasonable starting point: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, LoRA r=16, G=8,
lr 1e-5, max completion length 1024. A 16-24 GB GPU is enough for this size.
If you run out of memory: lower `--batch-size` (must stay divisible by `--num-generations`),
`--max-completion-length`, or use a smaller model.

## Before / after protocol (do this, it is the whole point)

```bash
python scripts/run_eval.py --provider hf --model Qwen/Qwen2.5-Coder-1.5B-Instruct --modes single --name base-before
python scripts/run_eval.py --provider hf --model outputs/grpo/merged               --modes single --name grpo-after
```

Same benchmark, same prompts, `single` mode, benchmark instances guaranteed disjoint
from the training instances. Report both tables honestly - including if the
result is flat or worse. With 40 tasks, treat differences under ~10 points as noise
and consider adding more tasks first.

## Things worth examining and writing up

* **Reward hacking:** read sampled completions. Is the model gaming the 0.4
  "feasible" floor with trivial code? Does the hidden term catch hard-coding?
* **Shaped vs binary reward:** `--binary-reward`; with a weak base model binary
  rewards give mostly zero-variance groups (no learning signal).
* **Generalisation:** train on 7 families, evaluate on the held-out 8th.
* **Learning curves:** mean reward and fraction of groups with zero variance per step.

## Not included (be honest about scope)

No RLHF: there is no preference data or learned reward model. Verifiable-reward RL
(RLVR/GRPO) is the appropriate technique when ground truth can be checked
programmatically, which is the case here.
