import random

import numpy as np
import torch


def seed_everything(seed: int, env) -> None:
    """Seed all random number generators for reproducibility.

    Note: env.reset(seed=seed) must still be called separately to seed
    the environment's initial state. This function seeds the action space
    for cases where we sample random actions (e.g., exploration).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.action_space.seed(seed)
