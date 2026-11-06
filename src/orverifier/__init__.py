"""OR-Verifier: solver-verified evaluation and RL rewards for LLM optimisation modelling."""

from .problems import FAMILIES, Instance, make_instance
from .verify import Verification, verify_code
from .rewards import reward_shaped, reward_binary, score_completion

__all__ = ["FAMILIES", "Instance", "make_instance", "Verification", "verify_code",
           "reward_shaped", "reward_binary", "score_completion"]
