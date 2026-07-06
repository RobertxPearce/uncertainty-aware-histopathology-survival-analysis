# Cox proportional-hazards partial likelihood loss.

import torch


def cox_loss(risk, time, event):
    """
    Cox negative partial log-likelihood loss. The loss is defined over
    the full dataset. Ties are not handled.
    
    :param risk: Tensor of shape [N] or [N,1]
                 Models predicted risk score
    :param time: Tensor of shape [N]
                 How long the patient was followed
    :param event: Tensor of shape [N]
                  Death 1 or alive/censored 0
    :return: Loss 
    """
    # Collapse a [N, 1] risk to [N] so the broadcasting below stays 1-D.
    risk = risk.reshape(-1)

    # Reorder the risk and events in descending order of time
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order].float()
    
    
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -((risk - log_cumsum) * event).sum()
    loss = loss / event.sum().clamp_min(1.0)
    
    return loss
    