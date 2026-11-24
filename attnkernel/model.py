"""Transductive attention network whose attention logits define the kernel.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedDotAttention(nn.Module):
    def __init__(self, d: int, heads: int, tau0: float = 10.0, attn_dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau0)))
        self.drop = nn.Dropout(attn_dropout)
        self.ln = nn.LayerNorm(d)

    def keys(self, H: torch.Tensor) -> torch.Tensor:
        """Unit-norm keys, shape (h, n, dh)."""
        # .contiguous() is required: batched matmul on transposed views is wrong on Apple MPS
        K = self.Wk(H).view(H.shape[0], self.h, self.dh).transpose(0, 1).contiguous()
        return F.normalize(K, dim=-1)

    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def logits(self, H: torch.Tensor, rows: slice) -> torch.Tensor:
        """S = tau K K^T for the given rows, shape (h, n_rows, n)."""
        K = self.keys(H)
        return self.tau() * K[:, rows].contiguous() @ K.transpose(1, 2).contiguous()

    def forward(self, H: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        n = H.shape[0]
        V = self.Wv(H).view(n, self.h, self.dh).transpose(0, 1).contiguous()   # (h, n, dh)
        chunk = chunk or n
        outs = []
        for s in range(0, n, chunk):
            rows = slice(s, min(n, s + chunk))
            S = self.logits(H, rows)                                            # (h, r, n)
            r = torch.arange(rows.start, rows.stop, device=H.device)
            S[:, r - rows.start, r] = float("-inf")                             # no self-attention
            A = self.drop(torch.softmax(S, dim=-1))
            outs.append((A @ V).transpose(0, 1).reshape(-1, self.h * self.dh))
        return self.ln(self.Wo(torch.cat(outs, 0)))


class AttentionKernelNet(nn.Module):
    """phi (linear, LayerNorm, GELU, dropout) -> one tied-QK attention layer -> one linear head per task."""

    def __init__(self, d_in: int, n_classes: dict[str, int], d: int = 128, heads: int = 4,
                 dropout: float = 0.3, attn_dropout: float = 0.1, tau0: float = 10.0):
        super().__init__()
        self.task_names = list(n_classes)
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.attn = TiedDotAttention(d, heads, tau0=tau0, attn_dropout=attn_dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(d, c) for t, c in n_classes.items()})
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, chunk: int | None = None) -> dict[str, torch.Tensor]:
        Z = self.attn(self.inp(X), chunk=chunk)
        return {t: self.heads[t](Z) for t in self.task_names}

    @torch.no_grad()
    def keys(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit-norm keys of all samples (h, N, dh) and the temperature tau (S = tau K K^T)."""
        return self.attn.keys(self.inp(X)), self.attn.tau()
