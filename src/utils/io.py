# Checkpoint save/load helpers.

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, val_cindex, model_config):
    """
    Persist enough state to resume or evaluate the model later.

    model_config is the dict of keyword arguments build_model() needs to rebuild
    the architecture.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_cindex": val_cindex,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": model_config,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    """
    Load a training checkpoint written by save_checkpoint.

    Returns the full dict as saved: model_state, optimizer_state, config
    (the model hyperparameters), epoch, and val_cindex.

    Kept model-agnostic on purpose: the caller rebuilds the model from
    checkpoint["config"] and then loads checkpoint["model_state"] into it, so
    this helper never needs to know which architecture produced the checkpoint.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
