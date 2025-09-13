# src/llamafactory/train/hidden_divergence.py
# Copyright 2025
# Utilities to compute hidden-state divergence D between Student and Teacher.
# D_l = (1 - CKA) + alpha_nmse * NMSE  (pooled per layer)
# D   = mean_l D_l
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "masked_mean_pool",
    "linear_cka",
    "nmse_normed",
    "HiddenDivergenceMeter",
]


# -----------------------------
# Basic pooling & similarity
# -----------------------------
def masked_mean_pool(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states along sequence dimension using attention_mask.

    Args:
        h: [B, T, D] hidden states
        attn_mask: [B, T] with 1 for valid tokens and 0 for padding (bool/long/float)
    Returns:
        pooled: [B, D]
    """
    assert h.dim() == 3, f"h must be [B,T,D], got {list(h.size())}"
    B, T, D = h.size()

    # Default to all-valid if mask is None
    if attn_mask is None:
        attn_mask = torch.ones(B, T, device=h.device, dtype=torch.long)

    # Shape checks & gentle fixups
    if attn_mask.dim() != 2:
        raise ValueError(f"attn_mask must be [B,T], got {list(attn_mask.size())}")
    if attn_mask.size(0) != B:
        raise ValueError(f"Batch mismatch: h={B} vs mask={attn_mask.size(0)}")
    if attn_mask.size(1) != T:
        if attn_mask.size(1) > T:
            attn_mask = attn_mask[:, :T]
        else:
            # Right-pad mask with ones to match T (treat missing as valid)
            pad = torch.ones(B, T - attn_mask.size(1), device=attn_mask.device, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, pad], dim=1)

    m = attn_mask.to(h.dtype).unsqueeze(-1)   # [B, T, 1]
    summed = (h * m).sum(dim=1)               # [B, D]
    denom = m.sum(dim=1).clamp_min(1.0)       # [B, 1]
    return summed / denom                     # [B, D]


@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Linear CKA between two batches of embeddings.

    Args:
        X, Y: [B, D]
    Returns:
        cka in [0, 1], scalar tensor
    """
    assert X.dim() == 2 and Y.dim() == 2, f"Expect [B,D], got {list(X.size())}, {list(Y.size())}"
    if X.size(0) != Y.size(0):
        B = min(X.size(0), Y.size(0))
        X, Y = X[:B], Y[:B]

    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    XtY = X.T @ Y
    num = (XtY ** 2).sum()
    den = torch.linalg.norm(X.T @ X, "fro") * torch.linalg.norm(Y.T @ Y, "fro")
    return (num / (den + eps)).clamp(0.0, 1.0)


def nmse_normed(S: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized MSE after L2-normalization on the feature dimension.

    Args:
        S, T: [B, D]
    Returns:
        scalar tensor
    """
    S = F.normalize(S, dim=-1, eps=eps)
    T = F.normalize(T, dim=-1, eps=eps)
    return F.mse_loss(S, T)


# -----------------------------
# Hidden divergence meter
# -----------------------------
class HiddenDivergenceMeter(nn.Module):
    """
    Compute hidden-state divergence D between Student and Teacher on selected layers.

    D_l = (1 - CKA) + alpha_nmse * NMSE    (pooled features per layer)
    D   = mean_l D_l

    Optionally projects S/T pooled features to a common low-dim (proj_dim) for cheaper CKA/NMSE.

    Usage:
        meter = HiddenDivergenceMeter(layers=None, proj_dim=256, pool="mean")
        out = meter(hS=list(student_out.hidden_states),
                    hT=list(teacher_out.hidden_states),
                    attn_mask=inputs["attention_mask"])
        # out -> {"D": scalar, "cka_mean": scalar, "nmse_mean": scalar, "per_layer": {lid: D_l}}
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        proj_dim: Optional[int] = 256,
        pool: str = "mean",              # "mean" or "last"
        alpha_nmse: float = 0.2,
    ):
        super().__init__()
        self.layers = layers
        self.proj_dim = proj_dim
        self.pool = pool
        self.alpha_nmse = alpha_nmse

        self.proj_S = nn.ModuleDict()
        self.proj_T = nn.ModuleDict()
        self._picked_layers: List[int] = []
        self._inited = False

    # -------- internal helpers --------
    def _pick_layers(self, n_layers: int) -> List[int]:
        # warn: hidden_states list often includes embeddings at index 0; we keep heuristics robust by sampling inside (1..n-2)
        if self.layers:
            return self.layers
        k = min(6, max(3, n_layers // 4))  # pick 3~6 layers
        # ensure valid range even for small n_layers
        lo = 1 if n_layers >= 3 else 0
        hi = max(0, n_layers - 2)
        if hi <= lo:
            # fallback: all layers
            return list(range(n_layers))
        idx = torch.linspace(lo, hi, steps=k).round().int().tolist()
        # deduplicate and sort
        return sorted(set(int(i) for i in idx if 0 <= int(i) < n_layers))

    def _maybe_build_projs(self, hS: List[torch.Tensor], hT: List[torch.Tensor]):
        if self._inited:
            return
        assert len(hS) > 0 and len(hT) > 0, "hidden_states list cannot be empty"
        device = hS[0].device

        # ensure same number of layers (common for HF CausalLM)
        if len(hS) != len(hT):
            n = min(len(hS), len(hT))
            hS = hS[:n]
            hT = hT[:n]

        self._picked_layers = self._pick_layers(len(hS))
        if self.proj_dim is None:
            self._inited = True
            return

        # create frozen linear projections
        for lid in self._picked_layers:
            dS = hS[lid].size(-1)
            dT = hT[lid].size(-1)

            linS = nn.Linear(dS, self.proj_dim, bias=False)
            linT = nn.Linear(dT, self.proj_dim, bias=False)
            linS.to(device); linT.to(device)

            # freeze projection weights — they are NOT trained
            for p in linS.parameters():
                p.requires_grad = False
            for p in linT.parameters():
                p.requires_grad = False

            self.proj_S[str(lid)] = linS
            self.proj_T[str(lid)] = linT

        self._inited = True

    # -------- main API --------
    @torch.no_grad()
    def forward(
        self,
        hS: List[torch.Tensor],          # each: [B, T, Ds]
        hT: List[torch.Tensor],          # each: [B, T, Dt]
        attn_mask: Optional[torch.Tensor],  # [B, T]
        use_last_token: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "D": scalar tensor,
                "cka_mean": scalar tensor,
                "nmse_mean": scalar tensor,
                "per_layer": {layer_id(int)-> scalar tensor}
            }
        """
        assert isinstance(hS, list) and isinstance(hT, list), "hS/hT should be lists of [B,T,D] tensors"
        assert len(hS) > 0 and len(hT) > 0, "empty hidden_states lists"
        assert hS[0].dim() == 3 and hT[0].dim() == 3, "hidden tensors must be [B,T,D]"

        # Align layer counts if needed
        if len(hS) != len(hT):
            n = min(len(hS), len(hT))
            hS = hS[:n]
            hT = hT[:n]

        # Init projections & pick layers
        self._maybe_build_projs(hS, hT)
        lids = self._picked_layers if self._picked_layers else list(range(len(hS)))

        # pooling mode
        pool_last = (use_last_token if use_last_token is not None else (self.pool == "last"))

        cka_vals: List[torch.Tensor] = []
        nmse_vals: List[torch.Tensor] = []
        d_per_layer: Dict[int, torch.Tensor] = {}

        for lid in lids:
            s = hS[lid]
            t = hT[lid]

            if pool_last:
                s_pool = s[:, -1, :]                      # [B, D_s]
                t_pool = t[:, -1, :]                      # [B, D_t]
            else:
                s_pool = masked_mean_pool(s, attn_mask)   # [B, D_s]
                t_pool = masked_mean_pool(t, attn_mask)   # [B, D_t]

            if self.proj_dim is not None:
                s_pool = self.proj_S[str(lid)](s_pool)    # [B, C]
                t_pool = self.proj_T[str(lid)](t_pool)    # [B, C]

            cka = linear_cka(s_pool, t_pool)              # scalar
            nmse = nmse_normed(s_pool, t_pool)            # scalar
            D_l = (1.0 - cka) + self.alpha_nmse * nmse    # scalar

            cka_vals.append(cka)
            nmse_vals.append(nmse)
            d_per_layer[int(lid)] = D_l

        cka_mean = torch.stack(cka_vals).mean()
        nmse_mean = torch.stack(nmse_vals).mean()
        D = torch.stack(list(d_per_layer.values())).mean()

        return {
            "D": D,
            "cka_mean": cka_mean,
            "nmse_mean": nmse_mean,
            "per_layer": d_per_layer,
        }
