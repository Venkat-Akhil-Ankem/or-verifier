import pytest

from orverifier.benchmark import build_benchmark
from orverifier.problems import FAMILIES, make_instance
from orverifier.reference_solutions import CORRECT, HEURISTIC
from orverifier.verify import (ERROR, INFEASIBLE, MALFORMED, OPTIMAL, REJECTED, SUBOPTIMAL,
                               TIMEOUT, verify_code)


def insts(fam, scale=0, n=3):
    return [make_instance(fam, 500 + k, scale) for k in range(n)]


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_reference_programs_are_optimal(fam):
    ver = verify_code(CORRECT[fam], fam, insts(fam, 1))
    assert ver.all_correct, [o.message for o in ver.outcomes]
    assert ver.failure_category() is None


def test_suboptimal_heuristic_is_detected():
    # greedy assignment is feasible but wrong on essentially every random instance
    ver = verify_code(HEURISTIC["assignment"], "assignment", insts("assignment", 1, 4))
    assert not ver.all_correct
    assert any(o.status == SUBOPTIMAL and o.rel_gap > 0 for o in ver.outcomes)


def test_gap_sign_for_max_and_min():
    ver = verify_code("def solve(data):\n    return []\n", "knapsack", insts("knapsack"))   # value 0 < optimum
    assert ver.visible.status == SUBOPTIMAL and ver.visible.rel_gap > 0
    ver = verify_code("def solve(data):\n    return [0] * len(data['times'])\n", "machine_scheduling",
                      insts("machine_scheduling"))                                           # everything on machine 0
    assert ver.visible.status == SUBOPTIMAL and ver.visible.rel_gap > 0


def test_statuses():
    i = insts("knapsack")
    assert verify_code("def solve(data):\n    return list(range(len(data['values'])))\n", "knapsack", i).visible.status == INFEASIBLE
    assert verify_code("def solve(data):\n    return 'banana'\n", "knapsack", i).visible.status == MALFORMED
    assert verify_code("def solve(data):\n    return [0, 0]\n", "knapsack", i).visible.status == MALFORMED
    assert verify_code("def solve(data):\n    raise ValueError('boom')\n", "knapsack", i).visible.status == ERROR
    assert verify_code("import os\ndef solve(data):\n    return []\n", "knapsack", i).visible.status == REJECTED
    assert verify_code("def solve(data):\n    while True: pass\n", "knapsack", i, timeout_s=1).visible.status == TIMEOUT


def test_hardcoded_answer_passes_visible_but_fails_hidden():
    vis = make_instance("knapsack", 900)
    others = [make_instance("knapsack", 901 + k) for k in range(3)]
    from orverifier.problems import FAMILIES
    sol = FAMILIES["knapsack"].reference(vis.data)[1]
    cheat = f"def solve(data):\n    return {sol}\n"
    ver = verify_code(cheat, "knapsack", [vis, *others])
    assert ver.visible_ok
    assert not ver.all_correct
    assert ver.failure_category() == "fails_hidden"
    assert ver.hidden_ok_frac < 1


def test_feedback_uses_visible_only_and_reveals_optimum_only_on_request():
    i = insts("assignment", 1, 3)
    ver = verify_code(HEURISTIC["assignment"], "assignment", i)
    if ver.visible.status == SUBOPTIMAL:
        assert ver.feedback(reveal_optimum=False) is None
        fb = ver.feedback(reveal_optimum=True)
        assert f"{ver.visible.optimum:.6g}" in fb
    err = verify_code("def solve(data):\n    return 1/0\n", "knapsack", insts("knapsack"))
    assert "ZeroDivisionError" in err.feedback()


def test_trivial_scale_larger_instances_have_unique_data():
    tasks = build_benchmark()
    keys = [str(i.data) for t in tasks for i in t.instances]
    assert len(keys) == len(set(keys)) == 40 * 4
