# SNGP variant of the ABMIL survival model (deterministic uncertainty).
#
# SNGP (Liu et al., "Simple and Principled Uncertainty Estimation with
# Deterministic Deep Learning via Distance Awareness", NeurIPS 2020) turns a
# standard network into a distance-aware one with a single forward pass. Two
# ingredients replace the plain encoder + linear risk head of ABMILSurvival:
#
#   1. Spectral normalisation on the encoder's linear layers (SpectralNormLinear)
#      soft-caps each layer's Lipschitz constant, so distances in feature space
#      track distances in input space (a bi-Lipschitz feature map). Points far
#      from the training data stay far in the pooled embedding.
#   2. A random-feature Gaussian process head (RandomFeatureGP) replaces the
#      linear risk head. Its posterior variance grows away from the training
#      data, giving a per-slide uncertainty in one deterministic pass, no
#      sampling (cf. MC-dropout) and no retraining (cf. deep ensembles).
#
# The GP mean is trained with the usual Cox loss through the shared train() loop
# (the head is linear in its random features, so gradients flow normally). The
# GP covariance is not learned by gradient descent: after training, one extra
# pass over the training data accumulates a Laplace precision matrix
# (fit_sngp_covariance in src/eval/uncertainty.py), which is inverted once to the
# covariance used at inference. Predicting then yields (risk_mean, risk_std) on
# the same Cox risk scale as predict(), so SNGP slots into the same reporting as
# the other uncertainty estimators.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abmil import ABMILEncoder


class SpectralNormLinear(nn.Linear):
    """
    Linear layer whose weight spectral norm is soft-capped at `coeff`.

    Estimates the largest singular value with power iteration (one step per
    forward is plenty once it has warmed up) and rescales the weight only when
    that estimate exceeds `coeff`, so well-behaved layers are left untouched.
    This bounds the layer's Lipschitz constant, which is what makes the SNGP
    feature extractor distance-aware.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.95,
        n_power_iterations=1,
        eps=1e-12,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        # Persisted so the power-iteration estimate carries across steps and
        # moves with the module under .to(device).
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0, eps=eps))
        self.register_buffer("v", F.normalize(torch.randn(in_features), dim=0, eps=eps))

    def _spectral_norm(self):
        weight = self.weight
        # Only refine u/v while training; at eval time reuse the converged pair
        # so the layer (and every downstream prediction) is deterministic.
        if self.training:
            with torch.no_grad():
                u, v = self.u, self.v
                for _ in range(self.n_power_iterations):
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps)
                    u = F.normalize(torch.mv(weight, v), dim=0, eps=self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        # sigma ~= u^T W v; kept in the graph so the cap regularises the weight.
        return torch.dot(self.u, torch.mv(weight, self.v))

    def forward(self, x):
        sigma = self._spectral_norm()
        # Shrink to spectral norm `coeff` only when the estimate overshoots it.
        scale = torch.clamp(self.coeff / (sigma + self.eps), max=1.0)
        return F.linear(x, self.weight * scale, self.bias)


class RandomFeatureGP(nn.Module):
    """
    Random-feature Gaussian process output head (the "NGP" in SNGP).

    Approximates an RBF-kernel GP with `num_features` random Fourier features
    phi(h) = sqrt(2 / D) * cos(input_scale * h W^T + b), where W and b are fixed
    random projections. A learnable linear layer over phi gives the GP mean
    (here the scalar Cox risk); the posterior covariance is a Laplace
    approximation whose precision is I * ridge + sum_i phi_i phi_i^T over the
    training data. Predictive variance at h is phi(h)^T Sigma phi(h), which grows
    where random features are unlike anything seen in training.

    The mean weights train by gradient descent through the Cox loss like any
    linear head. The precision is filled in afterwards by fit_sngp_covariance()
    and inverted once via compute_covariance(); only then is return_variance
    meaningful.
    """

    def __init__(self, in_features, num_features=1024, ridge_penalty=1.0, input_scale=None):
        super().__init__()
        self.num_features = num_features
        self.ridge_penalty = ridge_penalty
        # Default RBF length-scale ~ sqrt(in_features): the pooled embedding is
        # LayerNorm'd (~unit variance per dim), so this keeps the cosine argument
        # at O(1) scale regardless of embed_dim.
        self.input_scale = input_scale if input_scale is not None else in_features ** -0.5

        # Fixed random projection for the Fourier features (buffers, never trained).
        self.register_buffer("random_weight", torch.randn(num_features, in_features))
        self.register_buffer("random_bias", torch.rand(num_features) * 2.0 * torch.pi)

        # Learnable GP mean over the random features (the risk output).
        self.beta = nn.Linear(num_features, 1, bias=False)

        # Laplace precision (Sigma^-1) and its inverse. Start at the ridge prior;
        # compute_covariance() overwrites `covariance` once the precision is fit.
        eye = torch.eye(num_features)
        self.register_buffer("precision", ridge_penalty * eye.clone())
        self.register_buffer("covariance", eye.clone() / ridge_penalty)
        self.covariance_fitted = False

    def _random_features(self, h):
        projection = self.input_scale * F.linear(h, self.random_weight, self.random_bias)
        return (2.0 / self.num_features) ** 0.5 * torch.cos(projection)

    def reset_precision(self):
        """Reset the Laplace precision to the ridge prior before a fitting pass."""
        eye = torch.eye(self.num_features, device=self.precision.device)
        self.precision.copy_(self.ridge_penalty * eye)
        self.covariance_fitted = False

    @torch.no_grad()
    def update_precision(self, h):
        """Accumulate one batch's random-feature outer products into the precision."""
        phi = self._random_features(h)
        self.precision.add_(phi.t() @ phi)

    def compute_covariance(self):
        """Invert the accumulated precision once to get the posterior covariance."""
        self.covariance.copy_(torch.linalg.inv(self.precision))
        self.covariance_fitted = True

    def forward(self, h, return_variance=False):
        risk = self.beta(self._random_features(h)).squeeze(-1)
        if not return_variance:
            return risk
        if not self.covariance_fitted:
            raise RuntimeError(
                "GP covariance not fitted; call fit_sngp_covariance(model, train_loader) "
                "after training before requesting predictive variance."
            )
        phi = self._random_features(h)
        variance = ((phi @ self.covariance) * phi).sum(dim=-1)
        return risk, variance


class SNGPABMILSurvival(nn.Module):
    """
    ABMIL survival model with a spectral-normalised encoder and a GP risk head.

    A drop-in for ABMILSurvival in the training loop: forward(x, mask) returns
    the scalar Cox risk, and return_embedding=True adds the pooled embedding for
    SurvRNC. return_variance=True (used only after the covariance is fit) returns
    (risk, variance) for uncertainty-aware prediction.
    """

    def __init__(
        self,
        input_dim=1024,
        embed_dim=512,
        attention_dim=256,
        dropout=0.25,
        gated=True,
        hidden_dims=None,
        input_norm=False,
        pool_norm=True,
        num_features=1024,
        gp_ridge_penalty=1.0,
        gp_input_scale=None,
        spectral_norm_bound=0.95,
    ):
        super().__init__()
        sn_linear = partial(SpectralNormLinear, coeff=spectral_norm_bound)
        self.encoder = ABMILEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            gated=gated,
            hidden_dims=hidden_dims,
            input_norm=input_norm,
            pool_norm=pool_norm,
            linear=sn_linear,
        )
        self.risk_head = RandomFeatureGP(
            embed_dim,
            num_features=num_features,
            ridge_penalty=gp_ridge_penalty,
            input_scale=gp_input_scale,
        )

    def forward(self, x, mask=None, return_embedding=False, return_variance=False):
        """
        x: [B, N, D]; mask: [B, N] bool (True marks padding).

        returns:
        risk: [B]
        (risk, variance): when return_variance=True (needs a fit covariance)
        (risk, embedding): when return_embedding=True
        """
        h = self.encoder(x, mask=mask)
        if return_variance:
            return self.risk_head(h, return_variance=True)
        risk = self.risk_head(h)
        if return_embedding:
            return risk, h
        return risk


def build_sngp_model(
    input_dim=1024,
    embed_dim=512,
    attention_dim=256,
    dropout=0.25,
    gated=True,
    hidden_dims=None,
    input_norm=False,
    pool_norm=True,
    num_features=1024,
    gp_ridge_penalty=1.0,
    gp_input_scale=None,
    spectral_norm_bound=0.95,
):
    """
    Construct the SNGP ABMIL survival model.

    Keyword names match SNGPABMILSurvival's constructor so a model_config dict
    round-trips through the checkpoint config, mirroring build_model.
    """
    return SNGPABMILSurvival(
        input_dim=input_dim,
        embed_dim=embed_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        gated=gated,
        hidden_dims=hidden_dims,
        input_norm=input_norm,
        pool_norm=pool_norm,
        num_features=num_features,
        gp_ridge_penalty=gp_ridge_penalty,
        gp_input_scale=gp_input_scale,
        spectral_norm_bound=spectral_norm_bound,
    )
