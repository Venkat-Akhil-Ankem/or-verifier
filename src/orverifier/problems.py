"""
problems.py
-----------
Eight optimisation problem families with:

  * a random instance generator (two size scales: 0 = small, 1 = larger),
  * a *solution checker* (`evaluate`) that recomputes feasibility and the
    objective value from a returned solution - we never trust a number the
    model reports about itself,
  * an exact reference solver (`reference`, SciPy/HiGHS MILP) that gives the
    proven optimum.

The checker + reference pair is the "verifiable reward" used by both the
evaluation harness and the GRPO training code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

FEAS_TOL = 1e-5      # feasibility tolerance when checking constraints
OPT_TOL = 1e-6       # relative tolerance when comparing with the optimum


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Family:
    name: str
    sense: str                      # "min" or "max"
    statement: str                  # natural-language problem description
    data_spec: str                  # what `data` contains
    solution_spec: str              # what `solve(data)` must return
    generate: Callable[[random.Random, int], dict]
    evaluate: Callable[[dict, Any], tuple[bool, float, str]]
    reference: Callable[[dict], tuple[float, Any]]   # (optimum, a_solution)


@dataclass
class Instance:
    family: str
    data: dict
    optimum: float
    seed: int
    scale: int

    def to_dict(self) -> dict:
        return {"family": self.family, "data": self.data, "optimum": self.optimum,
                "seed": self.seed, "scale": self.scale}

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(d["family"], d["data"], d["optimum"], d["seed"], d["scale"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _milp(c, A=None, lo=None, hi=None, integrality=None, lb=0.0, ub=np.inf):
    """Minimise c.x subject to lo <= A x <= hi; raises unless proven optimal."""
    c = np.asarray(c, dtype=float)
    cons = []
    if A is not None and len(A) > 0:
        cons.append(LinearConstraint(np.asarray(A, dtype=float), lo, hi))
    integ = np.zeros(len(c)) if integrality is None else np.asarray(integrality)
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": 120})
    if res.status != 0:
        raise RuntimeError(f"reference MILP not solved to optimality "
                           f"(status={res.status}: {res.message})")
    return float(res.fun), res.x


def _ints(x, n, name, lo=None, hi=None, length=True) -> list[int]:
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    if length and len(x) != n:
        raise ValueError(f"{name} must have length {n}, got {len(x)}")
    out = []
    for v in x:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            raise ValueError(f"{name} must contain integers, got {v!r}")
        if abs(v - round(v)) > 1e-9:
            raise ValueError(f"{name} must contain integers, got {v!r}")
        v = int(round(v))
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"{name} value {v} outside [{lo}, {hi}]")
        out.append(v)
    return out


def _floats(x, n, name) -> list[float]:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise ValueError(f"{name} must be a list of length {n}")
    out = []
    for v in x:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v \
                or abs(v) == float("inf"):
            raise ValueError(f"{name} must contain finite numbers, got {v!r}")
        out.append(float(v))
    return out


def _index_set(x, n, name) -> list[int]:
    idx = _ints(x, n, name, 0, n - 1, length=False)
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name} contains duplicate indices")
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# 1. Knapsack (max)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_knapsack(rng, scale):
    n = rng.randint(8, 12) if scale == 0 else rng.randint(25, 40)
    w = [rng.randint(3, 30) for _ in range(n)]
    v = [rng.randint(5, 60) for _ in range(n)]
    return {"values": v, "weights": w, "capacity": sum(w) // 3}


def _eval_knapsack(d, sol):
    n = len(d["values"])
    idx = _index_set(sol, n, "selected items")
    wt = sum(d["weights"][i] for i in idx)
    val = float(sum(d["values"][i] for i in idx))
    if wt > d["capacity"] + FEAS_TOL:
        return False, val, f"total weight {wt} exceeds capacity {d['capacity']}"
    return True, val, "ok"


def _ref_knapsack(d):
    v = np.array(d["values"], float)
    w = np.array(d["weights"], float)
    f, x = _milp(-v, w[None, :], -np.inf, d["capacity"],
                 np.ones(len(v)), 0, 1)
    return -f, [i for i in range(len(v)) if x[i] > 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Assignment (min)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_assignment(rng, scale):
    n = rng.randint(4, 6) if scale == 0 else rng.randint(12, 20)
    return {"cost": [[rng.randint(1, 50) for _ in range(n)] for _ in range(n)]}


def _eval_assignment(d, sol):
    n = len(d["cost"])
    p = _ints(sol, n, "assignment", 0, n - 1)
    if sorted(p) != list(range(n)):
        return False, float("nan"), "assignment is not a permutation (each task exactly once)"
    return True, float(sum(d["cost"][i][p[i]] for i in range(n))), "ok"


def _ref_assignment(d):
    C = np.array(d["cost"], float)
    n = len(C)
    A = []
    for i in range(n):
        r = np.zeros((n, n)); r[i, :] = 1; A.append(r.ravel())
    for j in range(n):
        r = np.zeros((n, n)); r[:, j] = 1; A.append(r.ravel())
    f, x = _milp(C.ravel(), A, 1, 1, np.ones(n * n), 0, 1)
    X = x.reshape(n, n)
    return f, [int(np.argmax(X[i])) for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Transportation (min, LP)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_transport(rng, scale):
    m, n = (rng.randint(2, 3), rng.randint(3, 4)) if scale == 0 else \
           (rng.randint(5, 6), rng.randint(7, 9))
    supply = [rng.randint(10, 40) for _ in range(m)]
    total = sum(supply)
    cuts = sorted(rng.sample(range(1, total), n - 1))
    demand = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    cost = [[rng.randint(1, 20) for _ in range(n)] for _ in range(m)]
    return {"supply": supply, "demand": demand, "cost": cost}


def _eval_transport(d, sol):
    m, n = len(d["supply"]), len(d["demand"])
    if not isinstance(sol, (list, tuple)) or len(sol) != m:
        raise ValueError(f"flow must be a {m} x {n} nested list")
    F = [_floats(row, n, "flow row") for row in sol]
    if any(v < -FEAS_TOL for row in F for v in row):
        return False, float("nan"), "negative flow"
    for i in range(m):
        if abs(sum(F[i]) - d["supply"][i]) > FEAS_TOL * max(1, d["supply"][i]):
            return False, float("nan"), f"source {i} ships {sum(F[i]):.4g}, supply is {d['supply'][i]}"
    for j in range(n):
        s = sum(F[i][j] for i in range(m))
        if abs(s - d["demand"][j]) > FEAS_TOL * max(1, d["demand"][j]):
            return False, float("nan"), f"sink {j} receives {s:.4g}, demand is {d['demand'][j]}"
    return True, float(sum(d["cost"][i][j] * F[i][j] for i in range(m) for j in range(n))), "ok"


def _ref_transport(d):
    s, dm, C = d["supply"], d["demand"], np.array(d["cost"], float)
    m, n = C.shape
    A, rhs = [], []
    for i in range(m):
        r = np.zeros((m, n)); r[i, :] = 1; A.append(r.ravel()); rhs.append(s[i])
    for j in range(n):
        r = np.zeros((m, n)); r[:, j] = 1; A.append(r.ravel()); rhs.append(dm[j])
    f, x = _milp(C.ravel(), A, rhs, rhs)
    return f, x.reshape(m, n).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Production planning, integer (max)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_production(rng, scale):
    n, r = (rng.randint(3, 4), rng.randint(2, 3)) if scale == 0 else \
           (rng.randint(6, 8), rng.randint(4, 5))
    cap_hi = 120 if scale == 0 else 250
    return {"profit": [rng.randint(10, 50) for _ in range(n)],
            "usage": [[rng.randint(1, 9) for _ in range(n)] for _ in range(r)],
            "capacity": [rng.randint(60, cap_hi) for _ in range(r)]}


def _eval_production(d, sol):
    n = len(d["profit"])
    q = _ints(sol, n, "quantities", 0, None)
    for k, row in enumerate(d["usage"]):
        used = sum(row[j] * q[j] for j in range(n))
        if used > d["capacity"][k] + FEAS_TOL:
            return False, float("nan"), f"resource {k} used {used} > capacity {d['capacity'][k]}"
    return True, float(sum(d["profit"][j] * q[j] for j in range(n))), "ok"


def _ref_production(d):
    p = np.array(d["profit"], float)
    f, x = _milp(-p, d["usage"], -np.inf, d["capacity"], np.ones(len(p)), 0, np.inf)
    return -f, [int(round(v)) for v in x]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Uncapacitated facility location (min)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_facility(rng, scale):
    f, c = (rng.randint(3, 4), rng.randint(6, 8)) if scale == 0 else \
           (rng.randint(8, 10), rng.randint(20, 25))
    return {"fixed": [rng.randint(30, 80) for _ in range(f)],
            "cost": [[rng.randint(1, 30) for _ in range(f)] for _ in range(c)]}


def _eval_facility(d, sol):
    f, c = len(d["fixed"]), len(d["cost"])
    if not isinstance(sol, dict) or "open" not in sol or "assign" not in sol:
        raise ValueError('solution must be a dict with keys "open" and "assign"')
    opened = _index_set(sol["open"], f, "open")
    assign = _ints(sol["assign"], c, "assign", 0, f - 1)
    for i, j in enumerate(assign):
        if j not in opened:
            return False, float("nan"), f"customer {i} assigned to facility {j} which is not open"
    obj = sum(d["fixed"][j] for j in opened) + sum(d["cost"][i][assign[i]] for i in range(c))
    return True, float(obj), "ok"


def _ref_facility(d):
    fx, C = np.array(d["fixed"], float), np.array(d["cost"], float)
    c, m = C.shape
    nv = m + c * m
    obj = np.concatenate([fx, C.ravel()])
    A, lo, hi = [], [], []
    for i in range(c):
        r = np.zeros(nv); r[m + i * m: m + (i + 1) * m] = 1
        A.append(r); lo.append(1); hi.append(1)
    for i in range(c):
        for j in range(m):
            r = np.zeros(nv); r[m + i * m + j] = 1; r[j] = -1
            A.append(r); lo.append(-np.inf); hi.append(0)
    f, x = _milp(obj, A, lo, hi, np.ones(nv), 0, 1)
    X = x[m:].reshape(c, m)
    return f, {"open": [j for j in range(m) if x[j] > 0.5],
               "assign": [int(np.argmax(X[i])) for i in range(c)]}


# ──────────────────────────────────────────────────────────────────────────────
# 6. Identical-machine makespan scheduling (min)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_machine(rng, scale):
    n, m = (rng.randint(7, 9), rng.randint(2, 3)) if scale == 0 else \
           (rng.randint(12, 14), rng.randint(3, 4))
    return {"times": [rng.randint(2, 20) for _ in range(n)], "machines": m}


def _eval_machine(d, sol):
    n, m = len(d["times"]), d["machines"]
    a = _ints(sol, n, "machine per job", 0, m - 1)
    loads = [0] * m
    for j, k in enumerate(a):
        loads[k] += d["times"][j]
    return True, float(max(loads)), "ok"


def _ref_machine(d):
    p, m = np.array(d["times"], float), d["machines"]
    n = len(p)
    nv = n * m + 1
    obj = np.zeros(nv); obj[-1] = 1
    A, lo, hi = [], [], []
    for j in range(n):
        r = np.zeros(nv); r[j * m:(j + 1) * m] = 1
        A.append(r); lo.append(1); hi.append(1)
    for k in range(m):
        r = np.zeros(nv)
        for j in range(n):
            r[j * m + k] = p[j]
        r[-1] = -1
        A.append(r); lo.append(-np.inf); hi.append(0)
    integ = np.ones(nv); integ[-1] = 0
    ub = np.ones(nv); ub[-1] = np.inf
    lb = np.zeros(nv); lb[0] = 1          # symmetry breaking: job 0 on machine 0
    f, x = _milp(obj, A, lo, hi, integ, lb, ub)
    X = x[:-1].reshape(n, m)
    return f, [int(np.argmax(X[j])) for j in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Weighted set cover (min)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_setcover(rng, scale):
    u, s = (rng.randint(10, 14), rng.randint(8, 10)) if scale == 0 else \
           (rng.randint(30, 40), rng.randint(25, 30))
    subsets = [sorted(rng.sample(range(u), rng.randint(max(2, u // 5), max(3, u // 2))))
               for _ in range(s)]
    covered = set().union(*subsets)
    for e in range(u):
        if e not in covered:
            j = rng.randrange(s)
            subsets[j] = sorted(set(subsets[j]) | {e})
    return {"universe_size": u, "subsets": subsets,
            "costs": [rng.randint(1, 10) for _ in range(s)]}


def _eval_setcover(d, sol):
    s = len(d["subsets"])
    idx = _index_set(sol, s, "chosen subsets")
    covered = set()
    for j in idx:
        covered |= set(d["subsets"][j])
    missing = sorted(set(range(d["universe_size"])) - covered)
    cost = float(sum(d["costs"][j] for j in idx))
    if missing:
        return False, cost, f"elements not covered: {missing[:8]}"
    return True, cost, "ok"


def _ref_setcover(d):
    u, S, c = d["universe_size"], d["subsets"], np.array(d["costs"], float)
    s = len(S)
    A = np.zeros((u, s))
    for j, sub in enumerate(S):
        for e in sub:
            A[e, j] = 1
    f, x = _milp(c, A, 1, np.inf, np.ones(s), 0, 1)
    return f, [j for j in range(s) if x[j] > 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# 8. Open-pit block selection / ultimate pit limit (max)  - mining
# ──────────────────────────────────────────────────────────────────────────────

def _gen_pit(rng, scale):
    rows, cols = (3, rng.randint(4, 5)) if scale == 0 else (rng.randint(5, 6), rng.randint(7, 9))
    values, prec = [], []
    for r in range(rows):
        for c in range(cols):
            v = rng.randint(-25, 25) + (8 * r if rng.random() < 0.3 else 0)
            values.append(v)
            if r > 0:      # block needs the three blocks above it (45-degree slope)
                for dc in (-1, 0, 1):
                    if 0 <= c + dc < cols:
                        prec.append([r * cols + c, (r - 1) * cols + c + dc])
    return {"values": values, "precedence": prec}


def _eval_pit(d, sol):
    n = len(d["values"])
    idx = set(_index_set(sol, n, "selected blocks"))
    for i, j in d["precedence"]:
        if i in idx and j not in idx:
            return False, float("nan"), f"block {i} selected but required block {j} is not"
    return True, float(sum(d["values"][i] for i in idx)), "ok"


def _ref_pit(d):
    v = np.array(d["values"], float)
    n = len(v)
    P = d["precedence"]
    A = np.zeros((max(len(P), 1), n))
    for k, (i, j) in enumerate(P):
        A[k, i], A[k, j] = 1, -1
    f, x = _milp(-v, A, -np.inf, 0, np.ones(n), 0, 1)
    return -f, [i for i in range(n) if x[i] > 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family(
        "knapsack", "max",
        "Choose a subset of items to put in a knapsack. Each item has a value and a "
        "weight and can be chosen at most once. The total weight of the chosen items "
        "must not exceed the capacity. Maximise the total value.",
        'data["values"]: list[int]; data["weights"]: list[int]; data["capacity"]: int',
        "a list with the 0-based indices of the chosen items",
        _gen_knapsack, _eval_knapsack, _ref_knapsack),
    Family(
        "assignment", "min",
        "There are n workers and n tasks. cost[i][j] is the cost of giving task j to "
        "worker i. Every worker gets exactly one task and every task is done by exactly "
        "one worker. Minimise the total cost.",
        'data["cost"]: n x n list of lists of int',
        "a list p of length n where p[i] is the task (0-based) assigned to worker i",
        _gen_assignment, _eval_assignment, _ref_assignment),
    Family(
        "transportation", "min",
        "A product is shipped from sources to sinks. Source i has supply[i] units, sink "
        "j needs demand[j] units (total supply equals total demand). Shipping one unit "
        "from i to j costs cost[i][j]. Ship exactly the supply from every source and "
        "deliver exactly the demand to every sink at minimum total cost.",
        'data["supply"]: list[int]; data["demand"]: list[int]; data["cost"]: m x n list of lists',
        "an m x n nested list `flow` where flow[i][j] >= 0 is the number of units shipped from source i to sink j",
        _gen_transport, _eval_transport, _ref_transport),
    Family(
        "production_planning", "max",
        "A factory makes several products. One unit of product j earns profit[j] and "
        "uses usage[k][j] units of resource k. Resource k has capacity[k] available. "
        "Only whole units can be produced. Choose the production quantities to maximise "
        "total profit without exceeding any resource capacity.",
        'data["profit"]: list[int]; data["usage"]: r x n list of lists; data["capacity"]: list[int]',
        "a list of n non-negative integers, the quantity of each product",
        _gen_production, _eval_production, _ref_production),
    Family(
        "facility_location", "min",
        "Choose which facilities to open and assign every customer to exactly one open "
        "facility. Opening facility j costs fixed[j]. Serving customer i from facility "
        "j costs cost[i][j]. Minimise the total of opening costs and service costs "
        "(there are no capacity limits).",
        'data["fixed"]: list[int] (one per facility); data["cost"]: customers x facilities list of lists',
        'a dict {"open": [indices of opened facilities], "assign": [facility index for each customer]}',
        _gen_facility, _eval_facility, _ref_facility),
    Family(
        "machine_scheduling", "min",
        "Assign jobs to identical parallel machines. Job j takes times[j] time units on "
        "any machine; a machine processes its jobs one after another. Minimise the "
        "makespan, i.e. the finishing time of the busiest machine.",
        'data["times"]: list[int]; data["machines"]: int',
        "a list of length n where entry j is the machine (0-based) that runs job j",
        _gen_machine, _eval_machine, _ref_machine),
    Family(
        "set_cover", "min",
        "The elements 0..universe_size-1 must all be covered. Subset j contains the "
        "elements subsets[j] and costs costs[j] to select. Choose subsets so that every "
        "element is in at least one chosen subset, at minimum total cost.",
        'data["universe_size"]: int; data["subsets"]: list of lists of int; data["costs"]: list[int]',
        "a list with the indices of the chosen subsets",
        _gen_setcover, _eval_setcover, _ref_setcover),
    Family(
        "pit_block_selection", "max",
        "An open-pit mine is modelled as blocks. Mining block i changes profit by "
        "values[i] (negative means it costs money, e.g. waste rock). A pair [i, j] in "
        "precedence means block i can only be mined if block j is also mined (j lies "
        "above i). Select the set of blocks to mine so that all precedence relations "
        "hold and the total value is maximised. Mining nothing is allowed (value 0).",
        'data["values"]: list[int]; data["precedence"]: list of [i, j] pairs',
        "a list with the indices of the blocks to mine",
        _gen_pit, _eval_pit, _ref_pit),
]}


def make_instance(family: str, seed: int, scale: int = 0) -> Instance:
    """Deterministically generate an instance and solve it to proven optimality."""
    fam = FAMILIES[family]
    rng = random.Random(f"{family}|{scale}|{seed}")
    data = fam.generate(rng, scale)
    opt, _ = fam.reference(data)
    return Instance(family, data, float(opt), seed, scale)
