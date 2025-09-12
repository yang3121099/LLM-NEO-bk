# src/llamafactory/utils/hidden_divergence.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

def masked_mean_pool(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # h: [B,T,D], attn_mask: [B,T]  1=valid, 0=pad
    m = attn_mask.to(h.dtype).unsqueeze(-1)        # [B,T,1]
    denom = m.sum(dim=1).clamp_min(1.0)            # [B,1,1] before squeeze
    return (h * m).sum(dim=1) / denom.squeeze(2)   # [B,D]

@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # X, Y: [B,D]  (已池化)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XtY = X.T @ Y
    num = (XtY ** 2).sum()
    den = torch.linalg.norm(X.T @ X, "fro") * torch.linalg.norm(Y.T @ Y, "fro")
    return (num / (den + eps)).clamp(0.0, 1.0)

def nmse_normed(S: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    S = F.normalize(S, dim=-1, eps=eps)
    T = F.normalize(T, dim=-1, eps=eps)
    return F.mse_loss(S, T)

class HiddenDivergenceMeter(torch.nn.Module):
    """
    计算单个 batch 的分歧指标 D：
      对选定层 l：D_l = (1 - CKA) + 0.2 * NMSE
      D = mean_l D_l
    支持可选的低维投影，以降低计算/显存。
    """
    def __init__(self, layers: Optional[List[int]] = None, proj_dim: Optional[int] = 256):
        super().__init__()
        self.layers = layers  # None=训练时根据层数自动均匀采样
        self.proj_dim = proj_dim
        self.proj_S = torch.nn.ModuleDict()
        self.proj_T = torch.nn.ModuleDict()
        self._inited = False
        self._picked_layers: List[int] = []

    def _pick_layers(self, n_layers: int) -> List[int]:
        if self.layers:
            return self.layers
        # 均匀抽 3~6 层，足够稳定
        k = min(6, max(3, n_layers // 4))
        idx = torch.linspace(1, n_layers - 2, steps=k).round().int().tolist()
        return sorted(set(idx))

    def _maybe_build_projs(self, hS: List[torch.Tensor], hT: List[torch.Tensor]):
        if self._inited:
            return
        device = hS[0].device
        self._picked_layers = self._pick_layers(len(hS))
        if self.proj_dim is None:
            self._inited = True
            return
        for lid in self._picked_layers:
            dS = hS[lid].size(-1)
            self.proj_S[str(lid)] = torch.nn.Linear(dS, self.proj_dim, bias=False).to(device)
            dT = hT[lid].size(-1)
            self.proj_T[str(lid)] = torch.nn.Linear(dT, self.proj_dim, bias=False).to(device)
        self._inited = True

    @torch.no_grad()
    def forward(
        self,
        hS: List[torch.Tensor],   # 每层: [B,T,Ds]
        hT: List[torch.Tensor],   # 每层: [B,T,Dt]
        attn_mask: torch.Tensor,  # [B,T]
        use_last_token: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        返回:
            {"D": D标量, "cka_mean":..., "nmse_mean":..., "per_layer": {lid: D_l}}
        """
        self._maybe_build_projs(hS, hT)
        lids = self._picked_layers if self._picked_layers else range(len(hS))
        cka_vals, nmse_vals, d_per_layer = [], [], {}

        for lid in lids:
            s = hS[lid]
            t = hT[lid]
            if use_last_token:
                s_pool = s[:, -1, :]
                t_pool = t[:, -1, :]
            else:
                s_pool = masked_mean_pool(s, attn_mask)
                t_pool = masked_mean_pool(t, attn_mask)

            if self.proj_dim is not None:
                s_pool = self.proj_S[str(lid)](s_pool)
                t_pool = self.proj_T[str(lid)](t_pool)

            cka = linear_cka(s_pool, t_pool)
            nmse = nmse_normed(s_pool, t_pool)
            D_l = (1.0 - cka) + 0.2 * nmse
            cka_vals.append(cka)
            nmse_vals.append(nmse)
            d_per_layer[str(lid)] = D_l

        cka_mean = torch.stack(cka_vals).mean()
        nmse_mean = torch.stack(nmse_vals).mean()
        D = torch.stack(list(d_per_layer.values())).mean()
        return {"D": D, "cka_mean": cka_mean, "nmse_mean": nmse_mean, "per_layer": d_per_layer}
