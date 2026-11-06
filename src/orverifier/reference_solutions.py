"""
reference_solutions.py
----------------------
Hand-written `solve(data)` programs, as strings, in exactly the form an LLM is
asked to produce.

  CORRECT  - exact MILP formulations (SciPy/HiGHS) for every family.
             Used to (a) self-test the whole sandbox -> verifier chain and
             (b) give a "perfect model" for offline demos.
  HEURISTIC - plausible but suboptimal/incorrect programs, used to check that the
             verifier really catches wrong-but-feasible answers.
"""

CORRECT: dict[str, str] = {}
HEURISTIC: dict[str, str] = {}

_HEAD = "import numpy as np\nfrom scipy.optimize import milp, LinearConstraint, Bounds\n\n"

CORRECT["knapsack"] = _HEAD + '''
def solve(data):
    v = np.array(data["values"], float)
    w = np.array(data["weights"], float)
    res = milp(-v, constraints=LinearConstraint(w[None, :], -np.inf, data["capacity"]),
               integrality=np.ones(len(v)), bounds=Bounds(0, 1))
    return [i for i, x in enumerate(res.x) if x > 0.5]
'''

CORRECT["assignment"] = _HEAD + '''
def solve(data):
    C = np.array(data["cost"], float)
    n = len(C)
    A = []
    for i in range(n):
        r = np.zeros((n, n)); r[i, :] = 1; A.append(r.ravel())
    for j in range(n):
        r = np.zeros((n, n)); r[:, j] = 1; A.append(r.ravel())
    res = milp(C.ravel(), constraints=LinearConstraint(np.array(A), 1, 1),
               integrality=np.ones(n * n), bounds=Bounds(0, 1))
    X = res.x.reshape(n, n)
    return [int(np.argmax(X[i])) for i in range(n)]
'''

CORRECT["transportation"] = _HEAD + '''
def solve(data):
    s, d = data["supply"], data["demand"]
    C = np.array(data["cost"], float)
    m, n = C.shape
    A, rhs = [], []
    for i in range(m):
        r = np.zeros((m, n)); r[i, :] = 1; A.append(r.ravel()); rhs.append(s[i])
    for j in range(n):
        r = np.zeros((m, n)); r[:, j] = 1; A.append(r.ravel()); rhs.append(d[j])
    res = milp(C.ravel(), constraints=LinearConstraint(np.array(A), rhs, rhs),
               bounds=Bounds(0, np.inf))
    return res.x.reshape(m, n).tolist()
'''

CORRECT["production_planning"] = _HEAD + '''
def solve(data):
    p = np.array(data["profit"], float)
    U = np.array(data["usage"], float)
    cap = np.array(data["capacity"], float)
    res = milp(-p, constraints=LinearConstraint(U, -np.inf, cap),
               integrality=np.ones(len(p)), bounds=Bounds(0, np.inf))
    return [int(round(x)) for x in res.x]
'''

CORRECT["facility_location"] = _HEAD + '''
def solve(data):
    f = np.array(data["fixed"], float)
    C = np.array(data["cost"], float)
    c, m = C.shape
    nv = m + c * m
    obj = np.concatenate([f, C.ravel()])
    A, lo, hi = [], [], []
    for i in range(c):
        r = np.zeros(nv); r[m + i * m: m + (i + 1) * m] = 1
        A.append(r); lo.append(1); hi.append(1)
    for i in range(c):
        for j in range(m):
            r = np.zeros(nv); r[m + i * m + j] = 1; r[j] = -1
            A.append(r); lo.append(-np.inf); hi.append(0)
    res = milp(obj, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(nv), bounds=Bounds(0, 1))
    X = res.x[m:].reshape(c, m)
    return {"open": [j for j in range(m) if res.x[j] > 0.5],
            "assign": [int(np.argmax(X[i])) for i in range(c)]}
'''

CORRECT["machine_scheduling"] = _HEAD + '''
def solve(data):
    p = np.array(data["times"], float)
    n, m = len(p), data["machines"]
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
    lb = np.zeros(nv); lb[0] = 1            # symmetry breaking
    res = milp(obj, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=integ, bounds=Bounds(lb, ub))
    X = res.x[:-1].reshape(n, m)
    return [int(np.argmax(X[j])) for j in range(n)]
'''

CORRECT["set_cover"] = _HEAD + '''
def solve(data):
    u, S = data["universe_size"], data["subsets"]
    c = np.array(data["costs"], float)
    s = len(S)
    A = np.zeros((u, s))
    for j, sub in enumerate(S):
        for e in sub:
            A[e, j] = 1
    res = milp(c, constraints=LinearConstraint(A, 1, np.inf),
               integrality=np.ones(s), bounds=Bounds(0, 1))
    return [j for j in range(s) if res.x[j] > 0.5]
'''

CORRECT["pit_block_selection"] = _HEAD + '''
def solve(data):
    v = np.array(data["values"], float)
    n = len(v)
    P = data["precedence"]
    A = np.zeros((max(len(P), 1), n))
    for k, (i, j) in enumerate(P):
        A[k, i], A[k, j] = 1, -1
    res = milp(-v, constraints=LinearConstraint(A, -np.inf, 0),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    return [i for i in range(n) if res.x[i] > 0.5]
'''

# ── Heuristics: feasible but not always optimal ──────────────────────────────

HEURISTIC["knapsack"] = '''
def solve(data):
    v, w, cap = data["values"], data["weights"], data["capacity"]
    order = sorted(range(len(v)), key=lambda i: -v[i] / w[i])
    chosen = []
    for i in order:
        if w[i] <= cap:
            chosen.append(i)
            cap -= w[i]
    return chosen
'''

HEURISTIC["machine_scheduling"] = '''
def solve(data):
    loads = [0] * data["machines"]
    assign = [0] * len(data["times"])
    for j in sorted(range(len(data["times"])), key=lambda j: -data["times"][j]):
        k = loads.index(min(loads))
        assign[j] = k
        loads[k] += data["times"][j]
    return assign
'''

HEURISTIC["assignment"] = '''
def solve(data):
    C = data["cost"]
    used, out = set(), []
    for i in range(len(C)):
        j = min((j for j in range(len(C)) if j not in used), key=lambda j: C[i][j])
        used.add(j)
        out.append(j)
    return out
'''
