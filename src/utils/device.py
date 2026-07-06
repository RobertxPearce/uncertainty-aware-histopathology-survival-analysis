# Resolve the torch device for training and evaluation.
#
# One definition, imported by both the training and evaluation modules, so the
# device-selection policy lives in exactly one place instead of being copied.

import torch


def pick_device(requested="auto"):
    """
    Resolve a torch device. "auto" prefers CUDA, then Apple MPS, then CPU.

    Accepts either a string ("auto" | "cpu" | "cuda" | "mps") or an already
    resolved torch.device (returned unchanged).
    """
    if isinstance(requested, torch.device):
        return requested
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
