# Seed Python, NumPy, and Torch RNGs for reproducible runs.

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    Seed all RNGs we rely on and return the seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    Give each DataLoader worker its own NumPy/random seed.

    Workers are forked with the same state, so without this every worker
    draws the same random numbers in __getitem__.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """
    Seeded generator to pass to a DataLoader for reproducible shuffling.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
