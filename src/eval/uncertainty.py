# Uncertainty estimation for slide-level survival risk.
#
# Three predictive-uncertainty estimators that extend predict()'s per-slide frame
# with a risk_std column:
#   * mc_dropout_predict - stochastic forward passes through one trained model
#   * deep_ensemble_predict - spread across independently trained models
#   * sngp_predict - Gaussian-process posterior variance from one SNGP model
# All keep the raw Cox risk scale, so risk_mean plugs straight into the same
# C-index / Kaplan-Meier / calibration reporting as a point prediction, and
# risk_std is the accompanying uncertainty.

import numpy as np
import pandas as pd
import torch

from ..utils.device import pick_device
from .evaluate import predict


def _enable_dropout(model):
    """Re-enable dropout layers while the rest of the model stays in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, loader, device="auto", n_samples=100):
    """
    MC-dropout risk with predictive uncertainty.

    Keeps dropout active at inference and runs n_samples stochastic forward
    passes per slide, returning their mean and standard deviation. Returns a
    DataFrame with columns: case_id, slide_id, risk_mean, risk_std, time, event.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for an MC-dropout std.")

    device = pick_device(device)
    model.to(device)
    model.eval()
    _enable_dropout(model)

    records = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)

        samples = torch.stack(
            [model(features, mask=mask).reshape(-1) for _ in range(n_samples)], dim=0
        )  # [n_samples, B]
        risk_mean = samples.mean(dim=0).cpu().numpy()
        risk_std = samples.std(dim=0, unbiased=True).cpu().numpy()

        time = batch["time"].numpy().reshape(-1)
        event = batch["event"].numpy().reshape(-1)
        for i in range(len(risk_mean)):
            records.append(
                {
                    "case_id": batch["case_id"][i],
                    "slide_id": batch["slide_id"][i],
                    "risk_mean": float(risk_mean[i]),
                    "risk_std": float(risk_std[i]),
                    "time": float(time[i]),
                    "event": int(event[i]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["case_id", "slide_id", "risk_mean", "risk_std", "time", "event"],
    )


def deep_ensemble_predict(models, loader, device="auto"):
    """
    Deep-ensemble risk with predictive uncertainty.

    Scores the same slides with each independently trained model and summarizes
    across members. Returns a DataFrame with columns: case_id, slide_id,
    risk_mean, risk_std, time, event, where risk_std is the between-model
    disagreement.

    Assumes `loader` yields slides in a stable order across the members (an
    unshuffled evaluation loader does), so the per-slide predictions line up.
    """
    models = list(models)
    if len(models) < 2:
        raise ValueError("deep_ensemble_predict needs at least 2 models.")

    device = pick_device(device)
    frames = [predict(model, loader, device) for model in models]

    base = frames[0][["case_id", "slide_id", "time", "event"]].reset_index(drop=True)
    risks = np.stack([f["risk"].to_numpy() for f in frames], axis=0)  # [n_models, N]

    out = base.copy()
    out["risk_mean"] = risks.mean(axis=0)
    out["risk_std"] = risks.std(axis=0, ddof=1)
    return out[["case_id", "slide_id", "risk_mean", "risk_std", "time", "event"]]


@torch.no_grad()
def fit_sngp_covariance(model, loader, device="auto"):
    """
    Fit an SNGP model's GP covariance with one pass over the training loader.

    Run this once, after train() and before sngp_predict(): it resets the GP
    head's Laplace precision, accumulates the random-feature outer products over
    the given loader (use the *training* loader -- the covariance describes where
    the training data lives), then inverts the precision to the covariance. The
    model is modified in place and returned for convenience.
    """
    device = pick_device(device)
    model.to(device)
    model.eval()  # deterministic spectral norm + dropout off during the fit

    head = model.risk_head
    head.reset_precision()
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        embedding = model.encoder(features, mask=mask)
        head.update_precision(embedding)
    head.compute_covariance()
    return model


@torch.no_grad()
def sngp_predict(model, loader, device="auto"):
    """
    SNGP risk with predictive uncertainty from the GP posterior variance.

    Single deterministic forward pass per slide: no sampling and no ensemble.
    Requires the covariance to be fit first (see fit_sngp_covariance). Returns a
    DataFrame with columns: case_id, slide_id, risk_mean, risk_std, time, event,
    where risk_std is the square root of the GP posterior variance.
    """
    device = pick_device(device)
    model.to(device)
    model.eval()

    records = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)

        risk, variance = model(features, mask=mask, return_variance=True)
        risk_mean = risk.cpu().numpy().reshape(-1)
        risk_std = variance.clamp_min(0.0).sqrt().cpu().numpy().reshape(-1)

        time = batch["time"].numpy().reshape(-1)
        event = batch["event"].numpy().reshape(-1)
        for i in range(len(risk_mean)):
            records.append(
                {
                    "case_id": batch["case_id"][i],
                    "slide_id": batch["slide_id"][i],
                    "risk_mean": float(risk_mean[i]),
                    "risk_std": float(risk_std[i]),
                    "time": float(time[i]),
                    "event": int(event[i]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["case_id", "slide_id", "risk_mean", "risk_std", "time", "event"],
    )
