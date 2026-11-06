"""
llm.py
------
Thin LLM client layer so the agent does not care which model is behind it.

  AnthropicClient  - Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  HFClient         - a local Hugging Face model or a GRPO-trained checkpoint
  MockClient       - scripted/offline client for tests and demos
  CachedClient     - disk cache wrapper (reruns are free and reproducible)

All clients expose:  complete(messages, system, max_tokens, temperature) -> str
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence


class AnthropicClient:
    def __init__(self, model: str | None = None):
        import anthropic                                   # lazy import
        self.model = model or os.getenv("OR_MODEL", "claude-sonnet-5-5")
        self.client = anthropic.Anthropic()                # reads ANTHROPIC_API_KEY
        self.tag = f"anthropic:{self.model}"

    def complete(self, messages, system, max_tokens: int = 4096, temperature=None) -> str:
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class HFClient:
    """Local transformers model (also works for a LoRA/merged GRPO checkpoint)."""

    def __init__(self, model_path: str, max_new_tokens: int = 1536):
        import torch                                       # lazy imports
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.tag = f"hf:{model_path}"

    def complete(self, messages, system, max_tokens=None, temperature=None) -> str:
        msgs = [{"role": "system", "content": system}, *messages]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt").to(self.model.device)
        sample = temperature is not None and temperature > 0
        out = self.model.generate(
            ids, max_new_tokens=max_tokens or self.max_new_tokens, do_sample=sample,
            **({"temperature": temperature} if sample else {}))
        return self.tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)


class MockClient:
    """
    Offline client. `responses` is either a list (returned in order, last one
    repeats) or a function (messages, system) -> str.
    """

    def __init__(self, responses: Sequence[str] | Callable[[list, str], str]):
        self.responses = responses
        self.calls: list[tuple[list, str]] = []
        self.tag = "mock"

    def complete(self, messages, system, max_tokens=None, temperature=None) -> str:
        self.calls.append((list(messages), system))
        if callable(self.responses):
            return self.responses(messages, system)
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[i]


class CachedClient:
    """Cache completions on disk, keyed by (client tag, system, messages, sampling)."""

    def __init__(self, inner, cache_dir: str | Path = ".cache/llm"):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tag = getattr(inner, "tag", "client")

    def complete(self, messages, system, max_tokens=None, temperature=None) -> str:
        key = hashlib.sha256(json.dumps([self.tag, system, messages, max_tokens, temperature],
                                        sort_keys=True).encode()).hexdigest()
        f = self.dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())["text"]
        text = self.inner.complete(messages, system, max_tokens=max_tokens, temperature=temperature)
        f.write_text(json.dumps({"text": text}))
        return text
