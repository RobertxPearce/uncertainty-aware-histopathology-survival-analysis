# Survival metrics: concordance index and time-dependent AUC.
#
# Harrell's concordance index (C-index) is the standard discrimination metric
# for survival models and the signal we watch during training / use to select
# between MIL aggregators. It asks: of all pairs of patients whose outcomes we
# can actually order, how often does the model rank them the right way?

import numpy as np


def _to_numpy(x):
    """Accept a torch Tensor or anything array-like and return a 1-D float array."""
    if hasattr(x, "detach"):  # torch.Tensor -> cpu numpy without tracking grads
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def concordance_index(risk, time, event):
    """
    Harrell's concordance index for a risk score.

    Sign convention: risk is a hazard-like score where higher means worse
    prognosis / shorter survival (exactly what ABMILSurvival's Cox head and
    cox_loss produce). A pair is concordant when the patient the model scores as
    higher risk is the one who actually has the shorter survival time.

    A pair (i, j) is comparable only if the one who fails earlier is observed
    to fail (event == 1); pairs where the shorter time is censored can't be
    ordered and are skipped. Ties in risk on a comparable pair count as half.

    :param risk:  [N] predicted risk scores (torch Tensor or array-like).
    :param time:  [N] follow-up times.
    :param event: [N] event indicators, 1 = death/event, 0 = censored.
    :return: float in [0, 1]; 0.5 is random, 1.0 is perfect ranking. Returns
             float('nan') when there are no comparable pairs (e.g. a batch
             with no events), so callers can detect and skip it.
    """
    risk = _to_numpy(risk)
    time = _to_numpy(time)
    event = _to_numpy(event)

    if not (risk.shape == time.shape == event.shape):
        raise ValueError(
            f"risk/time/event must have the same length; got "
            f"{risk.shape}, {time.shape}, {event.shape}"
        )

    concordant = 0.0
    comparable = 0.0
    n = risk.shape[0]

    # Enumerate ordered pairs where i fails first (and is an observed event), so
    # each comparable pair is counted exactly once. i's shorter time is the
    # reference; j is anyone who was still at risk when i failed (time_j > time_i).
    for i in range(n):
        if event[i] != 1:
            continue
        later = time > time[i]  # j strictly outlived i -> orderable outcomes
        if not later.any():
            continue
        risk_j = risk[later]
        comparable += risk_j.size
        # i failed first, so the model is "right" when it scored i as higher risk.
        concordant += np.count_nonzero(risk[i] > risk_j)
        concordant += 0.5 * np.count_nonzero(risk[i] == risk_j)

    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)
