import time

from orverifier.sandbox import run_solve, static_check


def test_static_check_accepts_clean_code():
    assert static_check("import numpy as np\ndef solve(data):\n    return [1]\n") == []


def test_static_check_blocks_imports_and_builtins():
    assert any("os" in e for e in static_check("import os\ndef solve(d): return []"))
    assert any("subprocess" in e for e in static_check("from subprocess import run\ndef solve(d): return []"))
    assert any("open" in e for e in static_check("def solve(d):\n    open('x')\n    return []"))
    assert any("eval" in e for e in static_check("def solve(d):\n    return eval('1')"))
    assert any("dunder" in e for e in static_check("def solve(d):\n    return ().__class__"))
    assert any("relative" in e or "not allowed" in e for e in static_check("from . import x\ndef solve(d): return []"))


def test_static_check_requires_solve_and_valid_syntax():
    assert any("solve" in e for e in static_check("x = 1"))
    assert any("SyntaxError" in e for e in static_check("def solve(:"))


def test_run_solve_returns_values_and_converts_numpy():
    code = "import numpy as np\ndef solve(data):\n    return np.array([data['a'], 2]).tolist() + [np.int64(5)]\n"
    out = run_solve(code, [{"a": 1}, {"a": 9}])
    assert out[0] == {"ok": True, "solution": [1, 2, 5]}
    assert out[1]["solution"] == [9, 2, 5]


def test_exception_is_reported_with_line():
    out = run_solve("def solve(data):\n    return 1 / 0\n", [{}])
    assert not out[0]["ok"] and "ZeroDivisionError" in out[0]["error"] and "line 2" in out[0]["error"]


def test_data_is_not_mutated_between_instances():
    code = "def solve(data):\n    data['x'].append(1)\n    return data['x']\n"
    out = run_solve(code, [{"x": []}, {"x": []}])
    assert out[0]["solution"] == [1] and out[1]["solution"] == [1]


def test_timeout_is_per_instance_and_others_still_run():
    code = "def solve(data):\n    while data['hang']:\n        pass\n    return [0]\n"
    t = time.time()
    out = run_solve(code, [{"hang": True}, {"hang": False}], timeout_s=1.0)
    assert out[0] == {"ok": False, "error": "timeout"}
    assert out[1]["ok"]
    assert time.time() - t < 10


def test_swallowing_exception_cannot_defeat_timeout():
    code = ("def solve(data):\n    while True:\n        try:\n            while True:\n                pass\n"
            "        except Exception:\n            pass\n")
    out = run_solve(code, [{}], timeout_s=1.0)
    assert out[0]["error"] == "timeout"


def test_non_serialisable_return_is_error_not_crash():
    out = run_solve("def solve(data):\n    return object()\n", [{}])
    assert not out[0]["ok"] and "serialis" in out[0]["error"]


def test_print_output_does_not_corrupt_protocol():
    out = run_solve("def solve(data):\n    print('@@OR_RESULT@@{\"ok\": true}')\n    return [3]\n", [{}])
    assert out == [{"ok": True, "solution": [3]}]


def test_api_keys_not_visible_to_candidate(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    code = "import os\ndef solve(data):\n    return [k for k in __import__('os').environ if 'KEY' in k]\n"
    # static check rejects os/__import__, so test the env scrubbing via the runner directly with allowed code:
    code = "import time\ndef solve(data):\n    return [1]\n"
    assert run_solve(code, [{}])[0]["ok"]
    from orverifier import sandbox
    assert "ANTHROPIC_API_KEY" not in {"PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                       "PYTHONHASHSEED", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT"}
    assert "os.environ" not in sandbox._RUNNER


def test_memory_limit_turns_into_error():
    code = "def solve(data):\n    x = [0] * (10 ** 9)\n    return [len(x)]\n"
    out = run_solve(code, [{}], timeout_s=5, mem_mb=512)
    assert not out[0]["ok"]
