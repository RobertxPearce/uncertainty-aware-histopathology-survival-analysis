# SurvRNC rank-N-contrast loss for survival ordering.
#
# Rank-N-Contrast (Zha et al., NeurIPS 2023) adapted to survival prediction
# (Saeed et al., "SurvRNC", MICCAI 2024; arXiv:2403.10603, official code
# github.com/numanai/SurvRNC -> rnc_loss.py, class ProgRnCLoss). It orders the
# learned representation space by outcome: for an anchor patient, its embedding
# is pulled closer to patients with a similar survival time than to patients
# whose time-to-event is further away.
#
# This is an AUXILIARY loss. It shapes the embedding but produces no risk score,
# so it is combined with a primary survival loss (cox_loss here):
#     L = L_cox + beta * L_survrnc  (see survrnc_cox_loss; paper Eq. 4)
#
# Signature note: unlike cox_loss(risk, ...), this contrasts the pooled slide
# embedding, so it takes embeddings and the model must expose it
# (ABMILSurvival.forward(..., return_embedding=True)).

import torch


def survrnc_loss(embeddings, time, event, *, temperature=2.0, lambda_uncertain=0.5):
    """
    SurvRNC rank-N-contrast loss over slide embeddings, ranked by survival time.

    For anchor a and a positive p, the denominator runs over every other sample
    whose time-to-event is at least as far from a as p's is (the "harder or
    equal" set); minimising the loss makes the closer-in-time pair more similar
    than those. Right-censoring is handled with SurvRNC's comparability rule:
    each sample j is classed, relative to anchor a, as

      * CERTAIN  - the true ordering of j vs a is known: both had an event, or
                    a is censored but j had an earlier event (so a certainly
                    outlives j). A certain j farther than p is a hard negative
                    (weight 1); a certain j nearer than p is ignored.
      * UNCERTAIN - a censored time is only a lower bound, so the ordering is
                    unknown (a had the event and j is censored; both censored;
                    or a is censored and j's event is *later*). Such j enter
                    every positive's denominator with weight lambda_uncertain
                    regardless of their radius (paper's lambda; 0.5 was best).

    :param embeddings:      Tensor [N, D], pooled slide representations.
    :param time:            Tensor [N], observed follow-up time.
    :param event:           Tensor [N], 1 = event/death, 0 = censored.
    :param temperature:     Softmax temperature (paper default 2.0).
    :param lambda_uncertain: Weight for uncertain pairs, in [0, 1]. 0 drops them
                             entirely; 1 treats them as certain negatives.
    :return: Scalar loss (a differentiable 0 when no anchor has a usable pair).
    """
    n = embeddings.size(0)
    if n < 2:
        # Keep the tensor in the graph so callers can always add this term.
        return embeddings.sum() * 0.0

    time = time.reshape(-1).to(embeddings.dtype)
    event = event.reshape(-1)

    # Similarity = temperature-scaled negative Euclidean distance in embedding
    # space (paper's FeatureSimilarity('l2')). Subtract the per-row max (a self
    # distance of 0, detached) for numerical stability; it cancels in log(num) -
    # log(denom) so the loss is unchanged.
    sim = -torch.cdist(embeddings, embeddings, p=2) / temperature      # [N, N]
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = sim.exp()

    # Signed / absolute time gap from each anchor a to every sample j.
    d = time[None, :] - time[:, None]          # d[a, j] = t_j - t_a
    absd = d.abs()

    # Comparability class per (anchor a, sample j) - independent of the positive.
    ev = event.bool()
    both_events = ev[:, None] & ev[None, :]
    anchor_cens_j_event_earlier = (~ev[:, None]) & ev[None, :] & (d < 0)
    certain = both_events | anchor_cens_j_event_earlier               # [N, N]

    total = embeddings.new_zeros(())
    n_anchors = 0
    for a in range(n):
        absd_a = absd[a]
        valid = absd_a > 0                      # drop self and exact time ties
        if not valid.any():
            continue
        certain_a = certain[a] & valid
        uncertain_a = valid & ~certain[a]
        exp_a = exp_sim[a]

        # Uncertain samples weigh lambda in EVERY positive's denominator and do
        # not depend on the positive's radius, so it is one shared term/anchor.
        uncertain_term = lambda_uncertain * (uncertain_a.to(exp_a.dtype) * exp_a).sum()

        # Any sample with a real, known gap can act as a positive. For each such
        # p, the certain negatives are the certain samples strictly farther from
        # the anchor than p (paper's S_{a,p} intersected with the certain set).
        pos_idx = valid.nonzero(as_tuple=True)[0]                     # [P]
        farther = absd_a[None, :] > absd_a[pos_idx][:, None]          # [P, N]
        cert_mask = (farther & certain_a[None, :]).to(exp_a.dtype)    # [P, N]
        denom = cert_mask @ exp_a + uncertain_term                   # [P]

        keep = denom > 0                        # positives that have a negative
        if not keep.any():
            continue
        log_prob = sim[a, pos_idx][keep] - torch.log(denom[keep])
        total = total + (-log_prob).mean()      # mean over this anchor's positives
        n_anchors += 1

    if n_anchors == 0:
        return embeddings.sum() * 0.0
    return total / n_anchors                    # mean over valid anchors


def survrnc_cox_loss(
    risk, embedding, time, event,
    *, lambda_rnc=1.0, temperature=2.0, lambda_uncertain=0.5,
):
    """
    Combined primary + auxiliary loss with the training loop's 4-arg protocol:
        L = cox_loss(risk, time, event)
            + lambda_rnc * survrnc_loss(embedding, time, event)

    lambda_rnc is the paper's beta (weight between the two losses);
    lambda_uncertain is the paper's lambda (weight on uncertain pairs inside
    SurvRNC). Wire it up with functools.partial to fix the weights, e.g.
        loss_fn = partial(survrnc_cox_loss, lambda_rnc=0.5)
    """
    from .cox import cox_loss

    primary = cox_loss(risk, time, event)
    if lambda_rnc == 0.0:
        return primary
    return primary + lambda_rnc * survrnc_loss(
        embedding, time, event,
        temperature=temperature, lambda_uncertain=lambda_uncertain,
    )
