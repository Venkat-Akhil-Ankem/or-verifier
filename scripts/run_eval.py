"""
Evaluate an LLM on the OR benchmark.

Examples
--------
  # Claude via API (needs ANTHROPIC_API_KEY in .env or the environment)
  python scripts/run_eval.py --provider anthropic --model claude-sonnet-5-5

  # cheaper smoke run: 2 families, one task each
  python scripts/run_eval.py --provider anthropic --families knapsack set_cover --limit 1

  # a local / GRPO-trained model (needs torch + transformers)
  python scripts/run_eval.py --provider hf --model outputs/grpo-qwen-coder/merged

  # offline pipeline demo with a scripted "perfect model" (no API)
  python scripts/run_eval.py --provider mock

Outputs: results/<run-name>.jsonl, results/<run-name>.md, results/<run-name>.png
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orverifier.benchmark import load_benchmark  # noqa: E402
from orverifier.evaluate import run_benchmark, save_results, summarize  # noqa: E402
from orverifier.report import plot_by_family, to_markdown  # noqa: E402


def load_dotenv(path: Path) -> None:
    """Tiny .env reader (avoids a dependency)."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client(args):
    if args.provider == "anthropic":
        from orverifier.llm import AnthropicClient, CachedClient
        client = AnthropicClient(args.model)
        return CachedClient(client, ROOT / ".cache" / "llm") if not args.no_cache else client
    if args.provider == "hf":
        from orverifier.llm import HFClient
        return HFClient(args.model)
    # mock: a perfect model for the first call, used to demo the pipeline offline
    from orverifier.llm import MockClient
    from orverifier.reference_solutions import CORRECT

    def perfect(messages, system):
        first = messages[0]["content"]
        for fam, code in CORRECT.items():
            # the statement text identifies the family
            from orverifier.problems import FAMILIES
            if FAMILIES[fam].statement in first:
                return f"```python\n{code}\n```"
        return "no idea"
    return MockClient(perfect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "hf", "mock"], default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--modes", nargs="+", default=["single", "repair_errors", "repair_oracle"])
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max tasks per family")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--name", default=None, help="run name (default: provider-model)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    tasks = load_benchmark(ROOT / "benchmark" / "or_bench_v1.json")
    if args.families:
        tasks = [t for t in tasks if t.family in args.families]
    if args.limit:
        seen, keep = {}, []
        for t in tasks:
            if seen.get(t.family, 0) < args.limit:
                keep.append(t)
                seen[t.family] = seen.get(t.family, 0) + 1
        tasks = keep

    client = make_client(args)
    name = args.name or f"{args.provider}-{(args.model or 'default').replace('/', '_')}"
    print(f"Running {len(tasks)} tasks x modes {args.modes} with {name}")

    results = run_benchmark(tasks, client, modes=args.modes, max_rounds=args.max_rounds,
                            workers=args.workers, timeout_s=args.timeout,
                            temperature=args.temperature)
    out = ROOT / "results"
    save_results(results, out / f"{name}.jsonl")
    summary = summarize(results)
    (out / f"{name}.md").write_text(to_markdown(summary, f"Results: {name}"))
    plot_by_family(summary, out / f"{name}.png")
    print((out / f"{name}.md").read_text())
    print(f"Saved to results/{name}.jsonl|.md|.png")


if __name__ == "__main__":
    main()
