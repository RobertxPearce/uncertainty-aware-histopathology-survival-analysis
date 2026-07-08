# Train a MIL survival model.
#
# The loop is loss-agnostic: pass any survival loss with the signature
# loss_fn(risk, embedding, time, event) (cox_loss_step is the default). A MIL
# encoder pools each slide's patch-feature bag into one embedding, a linear risk
# head turns that into a scalar hazard score, and the loss fits those scores to
# the (time, event) labels. Losses that only need the risk (e.g. cox) ignore the
# embedding; representation losses (e.g. SurvRNC) use it. Validation watches
# Harrell's C-index to select the best epoch.
#
# train() receives prepared dataloaders, a model, and an optimizer. It does NOT
# discover files, split data, construct datasets, or create dataloaders -- that
# is the caller's job, so the loop stays reusable from a notebook. The CLI main()
# wires the stages together:
#     make_dataloaders_from_csv -> build_model -> build_optimizer -> train.
#
# Two things about the Cox loss shape the loop:
#   * The partial likelihood is defined over a risk set, and here that risk set
#     is a single batch, so prefer a large batch (see make_dataloaders).
#   * A batch with no events gives an undefined risk set, so we skip it.
#
# Example:
#   python -m src.train.train_loop \
#       --split-csv data/processed/splits.csv \
#       --epochs 50 --batch-size 32 --lr 1e-4 --out runs/cox_baseline

import argparse
from pathlib import Path

import numpy as np
import torch

from functools import partial

from ..data.dataset import make_dataloaders_from_csv
from ..eval.metrics import concordance_index
from ..losses.cox import cox_loss
from ..losses.survrnc import survrnc_cox_loss
from ..models.abmil import build_model
from ..utils.device import pick_device
from ..utils.io import save_checkpoint
from ..utils.seed import make_generator, seed_everything, worker_init_fn


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Loss protocol: the loop calls loss_fn(risk, embedding, time, event). Pure
# per-scalar losses like cox_loss(risk, time, event) ignore the embedding via
# this adapter; SurvRNC (survrnc_cox_loss) uses it. This keeps the loop
# loss-agnostic while supporting losses that need the representation.
def cox_loss_step(risk, embedding, time, event):
    """Adapter so the 3-arg cox_loss fits the 4-arg loop protocol."""
    return cox_loss(risk, time, event)


def build_optimizer(model, name="adamw", lr=1e-4, weight_decay=1e-5):
    """Construct the optimizer. Kept small and explicit; extend as needed."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r} (expected 'adam' or 'adamw').")


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=None):
    """
    One pass over the training loader. Returns the mean per-batch loss.

    Batches whose risk set has no events make a risk-set survival loss undefined,
    so we skip stepping on them and don't count them toward the reported loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        time = batch["time"].to(device)
        event = batch["event"].to(device)

        # No events in this batch => no comparable risk set => skip.
        if event.sum() == 0:
            continue

        risk, embedding = model(features, mask=mask, return_embedding=True)
        loss = loss_fn(risk, embedding, time, event)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device):
    """
    Score a split during training and return (c_index, mean_loss).

    The C-index is a ranking metric over the whole split, so we gather every
    slide's risk/time/event first and score once at the end rather than averaging
    per-batch C-indices (which would be biased on small batches).
    """
    model.eval()

    risks, times, events = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        time = batch["time"].to(device)
        event = batch["event"].to(device)

        risk, embedding = model(features, mask=mask, return_embedding=True)

        if event.sum() > 0:
            total_loss += loss_fn(risk, embedding, time, event).item()
            n_batches += 1

        risks.append(risk.detach().cpu())
        times.append(time.detach().cpu())
        events.append(event.detach().cpu())

    risk = torch.cat(risks)
    time = torch.cat(times)
    event = torch.cat(events)

    c_index = concordance_index(risk, time, event)
    mean_loss = total_loss / max(n_batches, 1)
    return c_index, mean_loss


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn=cox_loss_step,
    *,
    epochs=50,
    device="auto",
    early_stopping_patience=None,
    checkpoint_dir=None,
    model_config=None,
    grad_clip=None,
    verbose=True,
):
    """
    Train `model` on prepared dataloaders and return the run history.

    Receives everything it needs already built -- model, loaders, optimizer,
    loss_fn -- and never touches the filesystem for data. Selection is on the
    validation C-index (higher is better); on return the model holds the best
    epoch's weights.

    device: "auto" | "cpu" | "cuda" | "mps", or a torch.device.
    early_stopping_patience: stop after this many epochs without a new best
        C-index (None disables early stopping).
    checkpoint_dir: if given, writes best.pt (each new best) and last.pt (final
        epoch) there, and needs model_config so a checkpoint can rebuild the
        architecture; if None, train() does no disk I/O and just returns history.

    Returns a history dict:
        {"train_loss": [...], "val_loss": [...], "val_cindex": [...],
         "best_epoch": int, "best_cindex": float}
    """
    device = pick_device(device)
    model.to(device)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)

    history = {"train_loss": [], "val_loss": [], "val_cindex": []}
    best_cindex = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    last_epoch = 0
    last_val_cindex = float("nan")

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip=grad_clip
        )
        history["train_loss"].append(train_loss)

        msg = f"epoch {epoch:>3}/{epochs}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_cindex, val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
            last_val_cindex = val_cindex
            history["val_loss"].append(val_loss)
            history["val_cindex"].append(val_cindex)
            msg += f"  val_loss={val_loss:.4f}  val_cindex={val_cindex:.4f}"

            # Select on val C-index; nan (a split with no comparable pairs)
            # never beats a real score.
            if np.isfinite(val_cindex) and val_cindex > best_cindex:
                best_cindex = val_cindex
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
                if checkpoint_dir is not None:
                    save_checkpoint(
                        checkpoint_dir / "best.pt", model, optimizer,
                        epoch, val_cindex, model_config,
                    )
                msg += "  *"
            else:
                bad_epochs += 1

        if verbose:
            print(msg)

        if (
            early_stopping_patience is not None
            and val_loader is not None
            and bad_epochs >= early_stopping_patience
        ):
            if verbose:
                print(f"early stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

    # Persist the final-epoch weights before restoring the best, so last.pt is a
    # genuine "last epoch" checkpoint (useful for resuming) while the returned
    # in-memory model is the selected best.
    if checkpoint_dir is not None:
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer,
            last_epoch, last_val_cindex, model_config,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_cindex"] = best_cindex if np.isfinite(best_cindex) else float("nan")

    if verbose and val_loader is not None and np.isfinite(best_cindex):
        where = f"  ({checkpoint_dir / 'best.pt'})" if checkpoint_dir is not None else ""
        print(f"best val C-index: {best_cindex:.4f} at epoch {best_epoch}{where}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ABMIL + Cox head for slide-level survival."
    )
    parser.add_argument("--split-csv", type=Path, required=True,
                        help="Split metadata CSV produced by make_splits.")
    parser.add_argument("--out", type=Path, default=Path("runs/cox"),
                        help="Directory for best.pt / last.pt checkpoints.")

    # Optimisation
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Cox risk set = one batch; prefer large batches.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", default="adamw", help="adam | adamw")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Optional max-norm for gradient clipping.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop after N epochs without a new best val C-index.")
    parser.add_argument("--lambda-rnc", type=float, default=0.0,
                        help="Weight of the SurvRNC auxiliary loss (0 = pure Cox).")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="SurvRNC contrast temperature (used when --lambda-rnc > 0).")

    # Model
    parser.add_argument("--input-dim", type=int, default=1024,
                        help="Patch feature dim (UNI v1 = 1024).")
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--attention-dim", type=int, default=256,
                        help="Hidden size for ABMIL attention scoring.")
    parser.add_argument("--ungated-attention", dest="gated_attention",
                        action="store_false",
                        help="Use vanilla ABMIL attention instead of gated attention.")
    parser.add_argument("--dropout", type=float, default=0.25)

    # Data / runtime
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Cap patches per training bag (val/test always full).")
    parser.add_argument("--feature-key", default="features",
                        help="Dataset key holding patch features inside each .h5 bag.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-deterministic", dest="deterministic",
                        action="store_false",
                        help="Trade reproducibility for speed (cudnn benchmark).")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Root used to resolve relative feature paths.")
    return parser.parse_args()


def main():
    """CLI entry point: wire the stages, then hand prepared objects to train()."""
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = pick_device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader, _test_loader = make_dataloaders_from_csv(
        args.split_csv,
        batch_size=args.batch_size,
        feature_key=args.feature_key,
        max_patches=args.max_patches,
        project_root=args.project_root,
        num_workers=args.num_workers,
        generator=make_generator(args.seed),
        worker_init_fn=worker_init_fn,
    )
    if train_loader is None:
        raise ValueError(f"{args.split_csv} has no 'train' rows to train on.")

    model_config = dict(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        gated=args.gated_attention,
    )
    model = build_model(**model_config)
    optimizer = build_optimizer(
        model, name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay
    )

    # Pure Cox by default; add the SurvRNC representation term when asked.
    if args.lambda_rnc > 0.0:
        loss_fn = partial(
            survrnc_cox_loss,
            lambda_rnc=args.lambda_rnc,
            temperature=args.temperature,
        )
        print(f"Loss: Cox + {args.lambda_rnc} * SurvRNC (T={args.temperature})")
    else:
        loss_fn = cox_loss_step
        print("Loss: Cox")

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.out,
        model_config=model_config,
        grad_clip=args.grad_clip,
    )


if __name__ == "__main__":
    main()
