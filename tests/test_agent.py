"""Agent loop tests with a scripted MockClient (no network, no API key)."""
import json

from orverifier.agent import run_task
from orverifier.benchmark import build_benchmark
from orverifier.evaluate import load_results, run_benchmark, save_results, summarize
from orverifier.llm import CachedClient, MockClient
from orverifier.reference_solutions import CORRECT
from orverifier.report import to_markdown

TASKS = build_benchmark(families=["knapsack", "assignment"], n_small=1, n_large=1, n_hidden=2)


def fenced(code):
    return f"```python\n{code}\n```"


def knap_task():
    return next(t for t in TASKS if t.family == "knapsack" and t.scale == 0)


def test_repair_loop_fixes_a_crash():
    client = MockClient(["```python\ndef solve(data):\n    return 1/0\n```", fenced(CORRECT["knapsack"])])
    r = run_task(knap_task(), client, "repair_errors", max_rounds=3)
    assert [a.all_correct for a in r.attempts] == [False, True]
    assert r.attempts[0].failure == "runtime_error"
    assert r.final_correct and not r.first_correct and r.rounds_used == 2
    # the 2nd call must contain the error feedback
    second_prompt = client.calls[1][0]
    assert "ZeroDivisionError" in second_prompt[-1]["content"]
    assert second_prompt[-2]["role"] == "assistant"


def test_single_mode_never_repairs():
    client = MockClient(["```python\ndef solve(data):\n    return 1/0\n```", fenced(CORRECT["knapsack"])])
    r = run_task(knap_task(), client, "single")
    assert r.rounds_used == 1 and not r.final_correct and len(client.calls) == 1


def test_errors_mode_does_not_reveal_optimum_but_oracle_does():
    suboptimal = fenced("def solve(data):\n    return []\n")
    c1 = MockClient([suboptimal, fenced(CORRECT["knapsack"])])
    r1 = run_task(knap_task(), c1, "repair_errors")
    assert r1.rounds_used == 1                     # nothing actionable => loop stops
    c2 = MockClient([suboptimal, fenced(CORRECT["knapsack"])])
    r2 = run_task(knap_task(), c2, "repair_oracle")
    assert r2.final_correct and r2.rounds_used == 2
    assert "best possible objective" in c2.calls[1][0][-1]["content"]


def test_hidden_results_never_leak_into_feedback():
    t = knap_task()
    sol = __import__("orverifier.problems", fromlist=["FAMILIES"]).FAMILIES["knapsack"].reference(t.visible.data)[1]
    cheat = fenced(f"def solve(data):\n    return {sol}\n")
    client = MockClient([cheat, cheat])
    r = run_task(t, client, "repair_oracle", max_rounds=3)
    assert r.attempts[0].failure == "fails_hidden"
    assert r.rounds_used == 1                      # visible optimal => no feedback, hidden not revealed
    assert len(client.calls) == 1


def test_no_code_block_gets_feedback():
    client = MockClient(["Use dynamic programming.", fenced(CORRECT["knapsack"])])
    r = run_task(knap_task(), client, "repair_errors")
    assert r.attempts[0].failure == "no_code" and r.final_correct


def test_prompt_contains_example_but_not_optimum():
    client = MockClient([fenced(CORRECT["knapsack"])])
    t = knap_task()
    run_task(t, client, "single")
    prompt = client.calls[0][0][0]["content"]
    assert json.dumps(t.visible.data) in prompt
    assert '"optimum"' not in prompt            # the instance's stored optimum is never serialised into the prompt
    # the prompt builder does not even receive the optimum: only family + the instance's data
    from orverifier.prompts import build_user_prompt
    t.visible.optimum = -123456.789
    assert "-123456.789" not in build_user_prompt(t.family, t.visible)


def test_benchmark_runner_summary_and_report_roundtrip(tmp_path):
    def perfect(messages, system):
        from orverifier.problems import FAMILIES
        for fam, code in CORRECT.items():
            if FAMILIES[fam].statement in messages[0]["content"]:
                return fenced(code)
    res = run_benchmark(TASKS, MockClient(perfect), modes=("single", "repair_errors"), workers=2, progress=None)
    s = summarize(res)
    assert s["single"]["pass@1"] == 1.0 and s["repair_errors"]["final_acc"] == 1.0
    assert s["single"]["by_scale"] == {"small": 1.0, "large": 1.0}
    save_results(res, tmp_path / "r.jsonl")
    again = summarize(load_results(tmp_path / "r.jsonl"))
    assert again == s
    assert "pass@1" in to_markdown(s)


def test_cached_client_calls_inner_once(tmp_path):
    inner = MockClient(["a", "b"])
    c = CachedClient(inner, tmp_path)
    assert c.complete([{"role": "user", "content": "q"}], "sys") == "a"
    assert c.complete([{"role": "user", "content": "q"}], "sys") == "a"
    assert len(inner.calls) == 1
