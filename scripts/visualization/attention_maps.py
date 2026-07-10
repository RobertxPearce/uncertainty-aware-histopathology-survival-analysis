# Attention-map diagnostics for the trained ABMIL survival model.
#
# Answers two questions the eye alone can't:
#   1. WHERE does attention land?  -> a per-slide spatial heatmap rendered from
#      the patch `coords` stored in each feature bag (no WSI/openslide needed).
#   2. HOW concentrated is it?     -> attention entropy, effective number of
#      patches, and top-k mass. Diffuse attention (effective fraction near 1.0)
#      means the pooling is basically averaging and ABMIL isn't selecting signal.
#
# Example:
#   python scripts/visualization/attention_maps.py \
#       --checkpoint runs/abmil_cox_survrnc_uni_v2_TCGA_LUAD/best.pt \
#       --split-csv data/processed/experiments/uni_v2_luad/splits.csv \
#       --split test --n-slides 8

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import build_model, load_survival_table  # noqa: E402
from src.utils.io import load_checkpoint  # noqa: E402


def attention_stats(attn):
    """Concentration diagnostics for one attention vector (sums to 1, length N)."""
    a = np.asarray(attn, dtype=np.float64)
    a = a / a.sum()  # guard against fp drift
    n = a.size
    # Shannon entropy in nats; effective patch count = exp(H). A uniform bag has
    # effective == N; a bag that puts all mass on one patch has effective == 1.
    nz = a[a > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    effective = float(np.exp(entropy))
    order = np.sort(a)[::-1]
    top1_mass = float(order[: max(1, int(round(0.01 * n)))].sum())
    top5_mass = float(order[: max(1, int(round(0.05 * n)))].sum())
    return {
        "n_patches": n,
        "max_attn": float(a.max()),
        "effective_patches": effective,
        "effective_frac": effective / n,  # ~1.0 => uniform/averaging; near 0 => peaky
        "top1pct_mass": top1_mass,
        "top5pct_mass": top5_mass,
    }


def rasterize(coords, attn, step=None):
    """Place per-patch attention onto a regular grid inferred from coords.

    coords are level-0 pixel positions on a fixed patch stride, so integer-
    dividing by that stride recovers grid rows/cols. Returns a masked array with
    NaN off-tissue, ready for imshow.
    """
    coords = np.asarray(coords)
    xs, ys = coords[:, 0], coords[:, 1]
    if step is None:
        dx = np.diff(np.unique(xs))
        step = int(dx[dx > 0].min()) if dx.size else 1
    cols = ((xs - xs.min()) // step).astype(int)
    rows = ((ys - ys.min()) // step).astype(int)
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    grid[rows, cols] = attn
    return np.ma.masked_invalid(grid)


@torch.no_grad()
def slide_attention(model, feature_path, feature_key, device):
    """Full-bag forward for one slide -> (attention[N], risk, coords[N,2])."""
    with h5py.File(feature_path, "r") as h5:
        features = h5[feature_key][:]
        coords = h5["coords"][:] if "coords" in h5 else None
        step = int(h5[feature_key].attrs.get("patch_size_level0", 0)) or None
        if coords is not None and step is None:
            step = int(h5["coords"].attrs.get("patch_size_level0", 0)) or None
    x = torch.from_numpy(np.ascontiguousarray(features)).float().unsqueeze(0).to(device)
    risk, attention = model(x, return_attention=True)  # no mask: full bag
    return attention.squeeze(0).cpu().numpy(), float(risk.reshape(-1)[0]), coords, step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint from save_checkpoint (has config + model_state).")
    ap.add_argument("--split-csv", type=Path, required=True, help="Split CSV with feature_path/time/event/split columns.")
    ap.add_argument("--split", default="test", help="Which split column value to visualize (default: test).")
    ap.add_argument("--feature-key", default="features")
    ap.add_argument("--n-slides", type=int, default=8, help="How many slides to render (most-concentrated first).")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG (default: alongside the checkpoint).")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = build_model(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    table = load_survival_table(args.split_csv)
    rows = table[table["split"] == args.split].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"No rows with split == {args.split!r} in {args.split_csv}")

    records, panels = [], []
    for _, row in rows.iterrows():
        fp = Path(row["feature_path"])
        if not fp.is_file():
            print(f"  skip (missing bag): {fp}")
            continue
        attn, risk, coords, step = slide_attention(model, fp, args.feature_key, device)
        stats = attention_stats(attn)
        stats.update(slide_id=str(row.get("slide_id", fp.stem)), risk=risk)
        records.append(stats)
        panels.append((stats["slide_id"], coords, attn, step, stats))

    stats_df = pd.DataFrame(records).sort_values("effective_frac")  # most concentrated first
    print("\n=== attention concentration (sorted: most concentrated first) ===")
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(stats_df[["slide_id", "n_patches", "effective_patches", "effective_frac",
                        "top1pct_mass", "top5pct_mass", "max_attn", "risk"]].to_string(index=False))
    print(f"\nmean effective_frac = {stats_df['effective_frac'].mean():.3f}  "
          f"(near 1.0 = attention is ~uniform / just averaging; small = selective)")

    # --- render the most-concentrated N slides ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = stats_df["slide_id"].tolist()[: args.n_slides]
    picked = [p for sid in order for p in panels if p[0] == sid]
    n = len(picked)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (sid, coords, attn, step, st) in zip(axes.flat, picked):
        if coords is None:
            ax.set_title(f"{sid[:18]}\n(no coords)", fontsize=8)
            continue
        grid = rasterize(coords, attn, step)
        # Clip to the 99th percentile so a couple of hot patches don't wash out structure.
        vmax = np.percentile(attn, 99)
        ax.imshow(grid, cmap="inferno", vmax=vmax, interpolation="nearest")
        ax.invert_yaxis()  # image y grows downward
        ax.set_title(f"{sid[:22]}\neff={st['effective_frac']:.2f} top1%={st['top1pct_mass']:.2f}", fontsize=8)
    fig.tight_layout()

    out = args.out or args.checkpoint.with_name(f"{args.checkpoint.stem}_attention_{args.split}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    stats_df.to_csv(Path(out).with_suffix(".csv"), index=False)
    print(f"\nsaved figure -> {out}")
    print(f"saved stats  -> {Path(out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
