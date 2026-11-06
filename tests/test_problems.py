"""
Independent cross-check of the HiGHS reference optima by brute-force enumeration
on small instances, plus checker sanity tests. If the references or checkers were
wrong, every downstream number (evaluation AND RL reward) would be wrong.
"""
import itertools
import random

import pytest

from orverifier.problems import FAMILIES, make_instance

SEEDS = range(6)


def best(sense, vals):
    return min(vals) if sense == "min" else max(vals)


def test_knapsack_bruteforce():
    for s in SEEDS:
        inst = make_instance("knapsack", s)
        d, n = inst.data, len(inst.data["values"])
        vals = []
        for mask in range(1 << n):
            idx = [i for i in range(n) if mask >> i & 1]
            if sum(d["weights"][i] for i in idx) <= d["capacity"]:
                vals.append(sum(d["values"][i] for i in idx))
        assert max(vals) == pytest.approx(inst.optimum)


def test_assignment_bruteforce():
    for s in SEEDS:
        inst = make_instance("assignment", s)
        C, n = inst.data["cost"], len(inst.data["cost"])
        opt = min(sum(C[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        assert opt == pytest.approx(inst.optimum)


def test_transportation_bruteforce_tiny():
    # 2 sources x 3 sinks: row 0 determines row 1, so enumerate row 0 (integer data => integer optimum)
    from orverifier.problems import _ref_transport
    d = {"supply": [4, 5], "demand": [3, 4, 2], "cost": [[3, 1, 7], [2, 5, 4]]}
    opt = None
    for a in itertools.product(range(5), repeat=3):
        if sum(a) != 4:
            continue
        b = [d["demand"][j] - a[j] for j in range(3)]
        if min(b) < 0:
            continue
        c = sum(d["cost"][0][j] * a[j] + d["cost"][1][j] * b[j] for j in range(3))
        opt = c if opt is None else min(opt, c)
    assert _ref_transport(d)[0] == pytest.approx(opt)


def test_transportation_generated_balanced():
    for s in SEEDS:
        d = make_instance("transportation", s, 1).data
        assert sum(d["supply"]) == sum(d["demand"]) and min(d["demand"]) > 0


def test_production_bruteforce_tiny():
    from orverifier.problems import _ref_production
    d = {"profit": [7, 11], "usage": [[3, 5], [4, 2]], "capacity": [23, 19]}
    best_v = max(7 * a + 11 * b for a in range(30) for b in range(30)
                 if 3 * a + 5 * b <= 23 and 4 * a + 2 * b <= 19)
    assert _ref_production(d)[0] == pytest.approx(best_v)


def test_facility_bruteforce():
    for s in SEEDS:
        inst = make_instance("facility_location", s)
        fx, C = inst.data["fixed"], inst.data["cost"]
        f = len(fx)
        opt = min(sum(fx[j] for j in S) + sum(min(row[j] for j in S) for row in C)
                  for r in range(1, f + 1) for S in itertools.combinations(range(f), r))
        assert opt == pytest.approx(inst.optimum)


def test_machine_bruteforce():
    for s in SEEDS:
        inst = make_instance("machine_scheduling", s)
        p, m = inst.data["times"], inst.data["machines"]
        opt = min(max(sum(p[j] for j in range(len(p)) if a[j] == k) for k in range(m))
                  for a in itertools.product(range(m), repeat=len(p)))
        assert opt == pytest.approx(inst.optimum)


def test_setcover_bruteforce():
    for s in SEEDS:
        inst = make_instance("set_cover", s)
        d = inst.data
        S, c, u = d["subsets"], d["costs"], d["universe_size"]
        opt = min(sum(c[j] for j in range(len(S)) if mask >> j & 1)
                  for mask in range(1 << len(S))
                  if set().union(*[set(S[j]) for j in range(len(S)) if mask >> j & 1] or [set()])
                  >= set(range(u)))
        assert opt == pytest.approx(inst.optimum)


def test_pit_bruteforce():
    for s in SEEDS:
        inst = make_instance("pit_block_selection", s)
        d, n = inst.data, len(inst.data["values"])
        assert n <= 15
        vals = []
        for mask in range(1 << n):
            if all(not (mask >> i & 1) or (mask >> j & 1) for i, j in d["precedence"]):
                vals.append(sum(d["values"][i] for i in range(n) if mask >> i & 1))
        assert max(vals) == pytest.approx(inst.optimum)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_reference_solution_checks_out(name):
    """The solution returned by the reference solver must pass the checker with the same objective."""
    fam = FAMILIES[name]
    for scale in (0, 1):
        d = fam.generate(random.Random(f"x{scale}"), scale)
        opt, sol = fam.reference(d)
        ok, obj, msg = fam.evaluate(d, sol)
        assert ok, msg
        assert obj == pytest.approx(opt, abs=1e-6)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_generation_is_deterministic(name):
    assert make_instance(name, 7, 0).data == make_instance(name, 7, 0).data
    assert make_instance(name, 7, 0).data != make_instance(name, 8, 0).data


def test_checkers_reject_bad_solutions():
    k = make_instance("knapsack", 0).data
    ok, _, msg = FAMILIES["knapsack"].evaluate(k, list(range(len(k["values"]))))   # everything
    assert not ok and "capacity" in msg
    with pytest.raises(ValueError):
        FAMILIES["knapsack"].evaluate(k, [0, 0])                                   # duplicate
    with pytest.raises(ValueError):
        FAMILIES["knapsack"].evaluate(k, [999])                                    # out of range
    a = make_instance("assignment", 0).data
    ok, _, _ = FAMILIES["assignment"].evaluate(a, [0] * len(a["cost"]))            # not a permutation
    assert not ok
    p = make_instance("pit_block_selection", 0).data
    deep = max(range(len(p["values"])), key=lambda i: i)
    ok, _, msg = FAMILIES["pit_block_selection"].evaluate(p, [deep])               # skips precedence
    assert not ok and "required block" in msg
    f = make_instance("facility_location", 0).data
    ok, _, _ = FAMILIES["facility_location"].evaluate(f, {"open": [0], "assign": [1] * len(f["cost"])})
    assert not ok
