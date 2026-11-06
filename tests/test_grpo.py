"""Tests for the GRPO building blocks that CAN run without a GPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "grpo"))

from advantages import clipped_surrogate, group_advantages   # noqa: E402
from dataset import build_records                            # noqa: E402
from reward_fn import make_reward_fn                         # noqa: E402

from orverifier.benchmark import TRAIN_SEED_START, load_benchmark   # noqa: E402
from orverifier.reference_solutions import CORRECT                  # noqa: E402


def test_group_advantages_are_zero_mean_and_unit_scale_per_group():
    adv = group_advantages([0.1, 0.5, 0.9, 1.0, 0.0, 0.0, 1.0, 0.0], group_size=4).reshape(2, 4)
    assert np.allclose(adv.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(adv.std(axis=1), 1, atol=1e-2)
    assert adv[0].argmax() == 2 + 1 or adv[0].argmax() == 3          # best reward -> highest advantage


def test_group_with_identical_rewards_has_zero_advantage():
    assert np.allclose(group_advantages([0.3, 0.3, 0.3, 0.3], 4), 0)


def test_group_advantages_validates_shape():
    with pytest.raises(ValueError):
        group_advantages([1, 2, 3], 2)


def test_clipped_surrogate_clips_large_ratios():
    # positive advantage: ratio 2.0 is clipped to 1.2
    assert clipped_surrogate([np.log(2.0)], [0.0], [1.0], 0.2) == pytest.approx(1.2)
    # negative advantage: ratio 0.5 is clipped to 0.8 -> min(0.5*-1, 0.8*-1) = -0.8
    assert clipped_surrogate([np.log(0.5)], [0.0], [-1.0], 0.2) == pytest.approx(-0.8)
    # on-policy: ratio 1 -> mean advantage
    assert clipped_surrogate([0.0, 0.0], [0.0, 0.0], [1.0, -1.0]) == pytest.approx(0.0)


def test_dataset_records_have_expected_shape_and_no_benchmark_leakage():
    recs = build_records(per_family=2, n_hidden=2)
    assert len(recs) == 16
    bench = {json.dumps(i.data, sort_keys=True) for t in load_benchmark(ROOT / "benchmark" / "or_bench_v1.json")
             for i in t.instances}
    for r in recs:
        assert r["prompt"][0]["role"] == "system" and r["prompt"][1]["role"] == "user"
        insts = json.loads(r["instances"])
        assert len(insts) == 3 and all(i["seed"] >= TRAIN_SEED_START for i in insts)
        assert all(json.dumps(i["data"], sort_keys=True) not in bench for i in insts)
        assert json.dumps(insts[0]["data"]) in r["prompt"][1]["content"]


def test_reward_fn_scores_string_and_conversational_completions():
    recs = build_records(per_family=1, n_hidden=2)
    rec = next(r for r in recs if r["family"] == "knapsack")
    good = f"```python\n{CORRECT['knapsack']}\n```"
    bad = "```python\ndef solve(data):\n    return 1/0\n```"
    fn = make_reward_fn(workers=2)
    rewards = fn(prompts=[rec["prompt"]] * 3,
                 completions=[good, [{"role": "assistant", "content": good}], bad],
                 family=["knapsack"] * 3, instances=[rec["instances"]] * 3)
    assert rewards[0] == pytest.approx(1.0) and rewards[1] == pytest.approx(1.0)
    assert rewards[2] == pytest.approx(0.1)
    # and these rewards produce a meaningful GRPO signal
    adv = group_advantages(rewards + [0.0], 4)
    assert adv[0] > 0 > adv[2]


def test_reward_fn_never_raises_on_garbage():
    fn = make_reward_fn(workers=1)
    assert fn(completions=["```python\ndef solve(:\n```"], family=["knapsack"],
              instances=[json.dumps([])]) == [0.0]
