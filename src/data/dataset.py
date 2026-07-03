# Dataset and dataloader for the TRIDENT feature bags.
#
# Each slide is stored on disk as one .h5 "bag" holding a variable number of
# patch feature vectors (features: [N, D]) plus their coordinates (coords: [N, 2]).
# N differs from slide to slide, so a batch of bags cannot be stacked directly:
# collate_bags pads every bag in a batch up to the largest N and returns a
# boolean padding mask that TransMIL's attention uses to ignore the pad rows.
# The per-slide survival table (case_id, slide_id, feature_path, time, event,
# split) is the same one produced by make_survival_metadata / make_splits.

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .make_survival_metadata import load_survival_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_NAMES = ("train", "val", "test")


class SurvivalBagDataset(Dataset):
    """
    Serve one TRIDENT feature bag per slide, paired with its (time, event) label.

    Expects a metadata DataFrame with columns: feature_path, time, event and
    (optionally) slide_id / case_id. feature_path must resolve to a readable .h5
    file; use load_survival_metadata() to turn the relative paths in the CSV into
    absolute ones before constructing the dataset.

    Each item is a dict:
        features : FloatTensor [N, D]   patch features for the slide
        time     : FloatTensor []       follow-up time
        event    : FloatTensor []       1 death, 0 censored
        slide_id : str                  slide identifier (or "" if absent)
        case_id  : str                  patient identifier (or "" if absent)

    N varies per slide; batch with collate_bags to pad and build the mask.

    max_patches optionally caps N: bags with more patches are randomly
    subsampled (without replacement) each time they are drawn, which bounds
    memory and acts as light augmentation. Set it to None to always use every
    patch (the right choice for val/test).
    """

    def __init__(self, metadata, feature_key="features", max_patches=None, seed=None):
        for col in ("feature_path", "time", "event"):
            if col not in metadata.columns:
                raise ValueError(f"metadata is missing required column: {col!r}")

        self.metadata = metadata.reset_index(drop=True)
        self.feature_key = feature_key
        self.max_patches = max_patches
        # Per-dataset RNG so subsampling is reproducible and independent of global state.
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        feature_path = Path(row["feature_path"])
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Feature bag not found for row {idx}: {feature_path}"
            )

        # Open lazily inside __getitem__ so h5py handles stay inside the worker
        # process that reads them (opening in __init__ is not fork-safe).
        with h5py.File(feature_path, "r") as h5:
            if self.feature_key not in h5:
                raise KeyError(
                    f"{feature_path} has no dataset {self.feature_key!r}; "
                    f"found {list(h5.keys())}"
                )
            features = h5[self.feature_key][:]

        features = self._maybe_subsample(features)
        features = torch.from_numpy(np.ascontiguousarray(features)).float()

        return {
            "features": features,
            "time": torch.tensor(float(row["time"]), dtype=torch.float32),
            "event": torch.tensor(float(row["event"]), dtype=torch.float32),
            "slide_id": str(row.get("slide_id", "")),
            "case_id": str(row.get("case_id", "")),
        }

    def _maybe_subsample(self, features):
        """Randomly keep at most max_patches rows (no-op if unset or bag is small)."""
        n = features.shape[0]
        if self.max_patches is None or n <= self.max_patches:
            return features
        keep = self._rng.choice(n, size=self.max_patches, replace=False)
        keep.sort()  # preserve on-disk patch order; MIL is permutation-invariant anyway
        return features[keep]


def collate_bags(batch):
    """
    Pad a list of variable-length bags into a single batch for TransMIL.

    Returns a dict:
        features : FloatTensor [B, N_max, D] bags right-padded with zeros
        mask     : BoolTensor  [B, N_max] True marks padding to ignore
        time     : FloatTensor [B]
        event    : FloatTensor [B]
        slide_id : list[str] length B
        case_id  : list[str] length B

    The mask convention (True = pad) matches TransMILSurvival.forward(x, mask).
    """
    lengths = [item["features"].shape[0] for item in batch]
    n_max = max(lengths)
    dim = batch[0]["features"].shape[1]
    batch_size = len(batch)

    features = torch.zeros(batch_size, n_max, dim, dtype=torch.float32)
    mask = torch.ones(batch_size, n_max, dtype=torch.bool)  # start all-pad, unmask real rows
    for i, item in enumerate(batch):
        n = lengths[i]
        features[i, :n] = item["features"]
        mask[i, :n] = False

    return {
        "features": features,
        "mask": mask,
        "time": torch.stack([item["time"] for item in batch]),
        "event": torch.stack([item["event"] for item in batch]),
        "slide_id": [item["slide_id"] for item in batch],
        "case_id": [item["case_id"] for item in batch],
    }


def make_dataloaders(
    split_csv,
    batch_size=16,
    feature_key="features",
    max_patches=None,
    project_root=PROJECT_ROOT,
    num_workers=0,
    generator=None,
    worker_init_fn=None,
    splits=SPLIT_NAMES,
):
    """
    Build one DataLoader per split from a frozen split CSV.

    Reads the split table (case_id, slide_id, feature_path, time, event, split),
    resolves feature paths to absolute against project_root, and returns a dict
    mapping each requested split name to its DataLoader. The train loader is
    shuffled; val/test are not. max_patches is applied to train only, so
    evaluation always sees the full bag.

    Note on batch_size: the Cox partial likelihood is computed over the risk set
    *within a batch*, so very small batches give noisy gradients and a batch with
    no events contributes nothing. On small cohorts prefer a large batch (or set
    batch_size to the training-set size for full-batch Cox training).
    """
    metadata = load_survival_metadata(split_csv, project_root=project_root)
    if "split" not in metadata.columns:
        raise ValueError(f"{split_csv} has no 'split' column; run make_splits first.")

    loaders = {}
    for name in splits:
        rows = metadata[metadata["split"] == name]
        if rows.empty:
            continue
        is_train = name == "train"
        dataset = SurvivalBagDataset(
            rows,
            feature_key=feature_key,
            max_patches=max_patches if is_train else None,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            collate_fn=collate_bags,
            generator=generator if is_train else None,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )
    return loaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check the survival bag dataset: load one batch and print its shapes."
    )
    parser.add_argument("--split-csv", type=Path, required=True,
                        help="Split metadata CSV produced by make_splits.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Cap patches per training bag (default: keep all).")
    parser.add_argument("--feature-key", default="features",
                        help="Dataset key holding the patch features inside each .h5 bag.")
    return parser.parse_args()


def main():
    args = parse_args()
    loaders = make_dataloaders(
        args.split_csv,
        batch_size=args.batch_size,
        feature_key=args.feature_key,
        max_patches=args.max_patches,
    )
    for name, loader in loaders.items():
        batch = next(iter(loader))
        print(
            f"{name:<6} bags={len(loader.dataset):>4} "
            f"features={tuple(batch['features'].shape)} "
            f"mask={tuple(batch['mask'].shape)} "
            f"events={int(batch['event'].sum())}/{len(batch['event'])}"
        )


if __name__ == "__main__":
    main()
