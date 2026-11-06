"""
sandbox.py
----------
Run model-written Python in a separate process, with:

  * a static check (AST): import allow-list, banned builtins, no dunder tricks,
    and a required top-level `solve` function,
  * a subprocess with a scrubbed environment (API keys are NOT inherited),
  * per-instance wall-clock timeout (SIGALRM) + hard process timeout,
  * CPU / memory / file-size rlimits on POSIX.

SECURITY NOTE
This is a *guard rail against buggy or runaway generated code*, not a security
boundary against a determined attacker. Python sandboxes written in Python are
escapable. For untrusted models/code run the whole harness in a container
(see docs/design.md).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile

ALLOWED_IMPORTS = {
    "math", "itertools", "collections", "functools", "heapq", "bisect", "random",
    "typing", "fractions", "dataclasses", "operator", "copy", "time",
    "numpy", "scipy", "pulp",
}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
}

SENTINEL = "@@OR_RESULT@@"


# ──────────────────────────────────────────────────────────────────────────────
# Static check
# ──────────────────────────────────────────────────────────────────────────────

def static_check(code: str) -> list[str]:
    """Return a list of violations (empty list = code may be executed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{a.name}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                problems.append(f"import from '{node.module}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of '{node.id}' is not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"access to dunder attribute '{node.attr}' is not allowed (line {node.lineno})")

    if not any(isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body):
        problems.append("no top-level function `solve(data)` defined")
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Runner executed inside the child process
# ──────────────────────────────────────────────────────────────────────────────

_RUNNER = r'''
import sys, json, io, copy, signal, traceback
SENT = "%(sentinel)s"
real_out = sys.stdout

def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError("not JSON serialisable: " + type(o).__name__)

def emit(d):
    real_out.write(SENT + json.dumps(d) + "\n")
    real_out.flush()

class _Timeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _Timeout()

def _fmt():
    et, ev, tb = sys.exc_info()
    line = None
    for fr in traceback.extract_tb(tb):
        if fr.filename == "solution.py":
            line = fr.lineno
    s = (et.__name__ + ": " + str(ev))[:400]
    return s + (" (solution.py line %%d)" %% line if line else "")

code_path, limit = sys.argv[1], float(sys.argv[2])
instances = json.loads(sys.stdin.read())
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
have_alarm = hasattr(signal, "SIGALRM")
if have_alarm:
    signal.signal(signal.SIGALRM, _alarm)

ns = {"__name__": "candidate"}
try:
    with open(code_path) as fh:
        src = fh.read()
    exec(compile(src, "solution.py", "exec"), ns)
    solve = ns["solve"]
except BaseException:
    msg = _fmt()
    for _ in instances:
        emit({"ok": False, "error": "failed to load code: " + msg})
    sys.exit(0)

for data in instances:
    try:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, limit)
        sol = solve(copy.deepcopy(data))
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        sol = json.loads(json.dumps(sol, default=_default))
        emit({"ok": True, "solution": sol})
    except _Timeout:
        emit({"ok": False, "error": "timeout"})
    except BaseException:
        if have_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        emit({"ok": False, "error": _fmt()})
''' % {"sentinel": SENTINEL}


def _limits(cpu_s: int, mem_bytes: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 2))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))
    return apply


def run_solve(code: str, instances: list[dict], timeout_s: float = 10.0,
              mem_mb: int = 3072) -> list[dict]:
    """
    Execute `solve(data)` from `code` on every data dict (one subprocess).

    Returns one dict per instance:  {"ok": True, "solution": ...}  or
                                    {"ok": False, "error": "..."}  (error == "timeout" for timeouts)
    """
    posix = os.name == "posix"
    hard = timeout_s * len(instances) + 15
    with tempfile.TemporaryDirectory(prefix="orv_") as tmp:
        code_path = os.path.join(tmp, "solution.py")
        with open(code_path, "w") as fh:
            fh.write(code)
        runner_path = os.path.join(tmp, "_runner.py")
        with open(runner_path, "w") as fh:
            fh.write(_RUNNER)

        env = {
            "PATH": os.environ.get("PATH", ""),
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0", "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp,
        }
        if os.environ.get("SYSTEMROOT"):                       # needed on Windows
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

        proc = subprocess.Popen(
            [sys.executable, "-I", runner_path, code_path, str(timeout_s)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=tmp, env=env,
            start_new_session=posix,
            preexec_fn=_limits(int(hard), mem_mb << 20) if posix else None,
        )
        hard_timeout = False
        try:
            out, err = proc.communicate(json.dumps(instances), timeout=hard)
        except subprocess.TimeoutExpired:
            hard_timeout = True
            if posix:
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            out, err = proc.communicate()

    results: list[dict] = []
    for line in (out or "").splitlines():
        if line.startswith(SENTINEL):
            try:
                results.append(json.loads(line[len(SENTINEL):]))
            except json.JSONDecodeError:
                pass

    missing = len(instances) - len(results)
    if missing > 0:
        if hard_timeout:
            why = "timeout"
        elif proc.returncode and proc.returncode < 0:
            why = f"process killed by signal {-proc.returncode} (CPU/memory limit?)"
        else:
            why = f"process exited unexpectedly (code {proc.returncode}) {(err or '')[-200:]}"
        results += [{"ok": False, "error": why}] * missing
    return results[:len(instances)]
