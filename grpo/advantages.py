"""
advantages.py
-------------
The two formulas at the heart of GRPO, in plain NumPy so they can be unit-tested
and read without a GPU. (TRL's GRPOTrainer implements the same maths on tensors;
this file is the readable reference.)

Group-relative advantage - for each prompt, sample G completions, score them, and
compare each reward with its own group:

        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

No learned value function (critic) is needed - that is what makes GRPO cheaper
than PPO. Groups where every completion has the same reward get advantage 0
(nothing to learn from).

Clipped surrogate objective (per token, PPO-style):

        L = mean( min( rho * A , clip(rho, 1-e, 1+e) * A ) ),   rho = exp(logp_new - logp_old)
"""

from __future__ import annotations

import numpy as np


def group_advantages(rewards, group_size: int, eps: float = 1e-4) -> np.ndarray:
    """rewards: flat sequence of length n_prompts * group_size, grouped consecutively."""
    r = np.asarray(rewards, dtype=float)
    if r.size % group_size:
        raise ValueError("number of rewards must be a multiple of group_size")
    g = r.reshape(-1, group_size)
    adv = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + eps)
    return adv.ravel()


def clipped_surrogate(logp_new, logp_old, advantages, clip_eps: float = 0.2) -> float:
    rho = np.exp(np.asarray(logp_new, float) - np.asarray(logp_old, float))
    adv = np.asarray(advantages, float)
    return float(np.mean(np.minimum(rho * adv, np.clip(rho, 1 - clip_eps, 1 + clip_eps) * adv)))
