"""
train_grpo.py
-------------
GRPO with a solver-verified reward (RLVR) on a small code model, via Hugging Face
TRL + LoRA.

STATUS: written against the TRL GRPOTrainer API but NOT executed by the author
(no GPU available when this repo was built). The reward function, dataset
builder and advantage maths it relies on ARE unit-tested (tests/test_grpo.py).
Expect to adjust argument names to your installed TRL version (see requirements-train.txt).

Hardware: one 16-24 GB GPU for a 1.5B model with LoRA and 8 generations per prompt
(e.g. a rented RTX 4090 / A10G / Colab Pro L4). Start with --max-steps 50 to check it runs.

Usage
-----
  python grpo/dataset.py --per-family 150
  python grpo/train_grpo.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --max-steps 300
  python scripts/run_eval.py --provider hf --model outputs/grpo/merged --modes single --name grpo-after
  python scripts/run_eval.py --provider hf --model Qwen/Qwen2.5-Coder-1.5B-Instruct --modes single --name base-before
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "grpo"))

from reward_fn import make_reward_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "grpo" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "grpo"))
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8, help="group size G")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="per-device batch; must be divisible by --num-generations")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--binary-reward", action="store_true", help="0/1 reward instead of shaped")
    ap.add_argument("--no-lora", action="store_true")
    args = ap.parse_args()

    # Heavy imports only here so the rest of the repo works without a GPU stack.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    dataset = Dataset.from_list(rows)

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    config = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
        seed=0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=make_reward_fn(binary=args.binary_reward),
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)

    # Save a merged model so scripts/run_eval.py --provider hf can load it directly.
    try:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(args.out) / "merged")
        trainer.processing_class.save_pretrained(Path(args.out) / "merged")
        print("Merged model saved to", Path(args.out) / "merged")
    except Exception as e:                                    # full fine-tune, or older peft
        print("Skipped merge:", e)


if __name__ == "__main__":
    main()
