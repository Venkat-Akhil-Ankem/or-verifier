import pytest

from orverifier.problems import FAMILIES, make_instance
from orverifier.prompts import extract_code
from orverifier.reference_solutions import CORRECT
from orverifier.rewards import (WEIGHTS, reward_binary, reward_components, reward_shaped,
                                score_completion)
from orverifier.verify import verify_code


def I(fam="knapsack", n=3):
    return [make_instance(fam, 700 + k) for k in range(n)]


def fenced(code):
    return f"Here you go\n```python\n{code}\n```\n"


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_code_gets_full_reward():
    assert score_completion(fenced(CORRECT["knapsack"]), "knapsack", I()) == pytest.approx(1.0)
    assert score_completion(fenced(CORRECT["knapsack"]), "knapsack", I(), binary=True) == 1.0


def test_reward_ordering_matches_quality():
    ins = I("knapsack", 4)
    perfect = score_completion(fenced(CORRECT["knapsack"]), "knapsack", ins)
    suboptimal = score_completion(fenced("def solve(data):\n    return []\n"), "knapsack", ins)
    infeasible = score_completion(fenced("def solve(data):\n    return list(range(len(data['values'])))\n"), "knapsack", ins)
    crash = score_completion(fenced("def solve(data):\n    return 1/0\n"), "knapsack", ins)
    rejected = score_completion(fenced("import os\ndef solve(data):\n    return []\n"), "knapsack", ins)
    nocode = score_completion("I think you should use dynamic programming.", "knapsack", ins)
    assert perfect > suboptimal > infeasible > crash > rejected
    assert rejected == nocode == 0.0


def test_exact_values():
    ins = I("knapsack", 4)
    assert score_completion(fenced("def solve(data):\n    return []\n"), "knapsack", ins) == pytest.approx(0.4)   # fmt+run+feasible
    assert score_completion(fenced("def solve(data):\n    return 1/0\n"), "knapsack", ins) == pytest.approx(0.1)  # fmt only


def test_hidden_term_is_anti_memorisation():
    vis = make_instance("knapsack", 900)
    others = [make_instance("knapsack", 901 + k) for k in range(3)]
    sol = FAMILIES["knapsack"].reference(vis.data)[1]
    r = score_completion(fenced(f"def solve(data):\n    return {sol}\n"), "knapsack", [vis, *others])
    assert r < 1.0
    assert r >= 0.7           # got the visible one right
    assert reward_binary(verify_code(f"def solve(data):\n    return {sol}\n", "knapsack", [vis, *others])) == 0.0


def test_components_zero_when_none():
    assert sum(reward_components(None).values()) == 0.0
    assert reward_shaped(None) == 0.0


def test_extract_code():
    assert extract_code("x\n```python\ndef solve(d):\n    return 1\n```\n").startswith("def solve")
    assert extract_code("```\ndef solve(d): pass\n```") is not None
    assert extract_code("```python\nprint(1)\n```\n```python\ndef solve(d): pass\n```").startswith("def solve")
    assert extract_code("def solve(d):\n    return 1").startswith("def solve")
    assert extract_code("no code at all") is None
