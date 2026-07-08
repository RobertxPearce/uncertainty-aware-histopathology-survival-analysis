# Attention-based multiple-instance learning for whole-slide feature bags.
#
# ABMIL pools patch embeddings with learned per-patch attention scores. Unlike a
# vanilla Transformer, this is linear in the number of patches, so it can score
# large histopathology bags without forming an N x N attention matrix.

import torch
import torch.nn as nn


class ABMILEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        embed_dim=512,
        attention_dim=256,
        dropout=0.25,
        gated=True,
        linear=nn.Linear,
    ):
        super().__init__()

        self.gated = gated

        # linear is the layer factory for the encoder's linear maps. It
        # defaults to nn.Linear; SNGP passes a spectral-normalised variant so the
        # feature extractor stays distance-aware (see src/models/sngp.py).
        self.patch_proj = nn.Sequential(
            linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_v = linear(embed_dim, attention_dim)
        if gated:
            self.attention_u = linear(embed_dim, attention_dim)
        else:
            self.attention_u = None
        self.attention_w = linear(attention_dim, 1)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, return_attention=False):
        """
        x: [B, N, D]
           B = batch size
           N = number of patches
           D = patch feature dimension
        mask: [B, N] bool, optional
           True marks padding patches that should receive zero attention.

        returns:
        h: [B, embed_dim]
        attention: [B, N], optional when return_attention=True
        """
        if x.dim() != 3:
            raise ValueError(f"ABMILEncoder expects x with shape [B, N, D], got {tuple(x.shape)}")

        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match x batch/patch dims {tuple(x.shape[:2])}"
                )
            mask = mask.bool()
            if mask.all(dim=1).any():
                raise ValueError("ABMILEncoder received a bag with no unmasked patches.")

        # [B, N, D] -> [B, N, embed_dim]
        h = self.patch_proj(x)

        # Ilse et al. gated attention: w^T(tanh(Vh) * sigmoid(Uh)).
        attention_features = torch.tanh(self.attention_v(h))
        if self.gated:
            attention_features = attention_features * torch.sigmoid(self.attention_u(h))

        logits = self.attention_w(attention_features).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        attention = torch.softmax(logits, dim=1)
        if mask is not None:
            attention = attention.masked_fill(mask, 0.0)

        # Weighted sum over patches: [B, 1, N] x [B, N, embed_dim] -> [B, embed_dim].
        pooled = torch.bmm(attention.unsqueeze(1), h).squeeze(1)
        pooled = self.norm(pooled)

        if return_attention:
            return pooled, attention
        return pooled


class ABMILSurvival(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        embed_dim=512,
        attention_dim=256,
        dropout=0.25,
        gated=True,
    ):
        super().__init__()

        self.encoder = ABMILEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            gated=gated,
        )
        self.risk_head = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None, return_attention=False, return_embedding=False):
        """
        x: [B, N, D]
        mask: [B, N] bool, optional, True marks padding patches.

        returns:
        risk: [B]
        attention: [B, N], optional when return_attention=True
        embedding: [B, embed_dim], optional when return_embedding=True

        When both flags are set the order is (risk, attention, embedding). The
        embedding is the pooled slide representation that feeds the risk head;
        SurvRNC contrasts on it (see src/losses/survrnc.py).
        """
        if return_attention:
            h, attention = self.encoder(x, mask=mask, return_attention=True)
        else:
            h = self.encoder(x, mask=mask)
            attention = None

        risk = self.risk_head(h).squeeze(-1)

        if return_attention and return_embedding:
            return risk, attention, h
        if return_attention:
            return risk, attention
        if return_embedding:
            return risk, h
        return risk


def build_model(input_dim=1024, embed_dim=512, attention_dim=256, dropout=0.25, gated=True):
    """
    Construct the ABMIL encoder + Cox risk head.

    The single model-construction path used by both training and evaluation.
    Keyword names match ABMILSurvival's constructor, so a model_config dict
    round-trips through save_checkpoint / load_model unchanged.
    """
    return ABMILSurvival(
        input_dim=input_dim,
        embed_dim=embed_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        gated=gated,
    )
