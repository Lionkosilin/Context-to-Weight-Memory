"""The best rank-r ΔW for one layer's keys and targets under the generic-text metric.

    J(ΔW) = ‖ΔW K − V‖²_F + λ tr(ΔW Σ ΔWᵀ),      λ = ridge · E[zᵀ Σ⁻¹ z]

K (d_in x n) holds keys, V (d_out x n) holds the output change each key must produce, and Σ is the
key second moment on generic text (ctw.stats). A generic key k has kᵀ Σ⁻¹ k ≈ E[zᵀ Σ⁻¹ z], so with
this λ it keeps a fraction 1/(1 + ridge) of its target. The minimizer is

    ΔW* = V Kᵀ (K Kᵀ + λ Σ)⁻¹ = V (G + λI)⁻¹ Kᵀ Σ⁻¹,      G = Kᵀ Σ⁻¹ K.

J is quadratic with Hessian M = K Kᵀ + λ Σ, so J(ΔW) − J(ΔW*) = ‖(ΔW − ΔW*) M^{1/2}‖²_F and the best
rank-r write is the truncated SVD in that metric: ΔW_r = [ΔW* M^{1/2}]_r M^{-1/2}. With
G = E diag(g) Eᵀ and V E diag(√(g/(g+λ))) = P S Rᵀ, that is

    A = P_r S_r,      B = R_rᵀ diag(1/√(g(g+λ))) Eᵀ Kᵀ Σ⁻¹,

and every step after Σ⁻¹K works on n x n matrices.
"""

from __future__ import annotations

import torch

from .stats import KeyStats


def covariance_ridge(keys: torch.Tensor, values: torch.Tensor, stats: KeyStats, ridge: float,
                     rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return A (d_out x r), B (r x d_in), and fit diagnostics."""
    k, v = keys.double().cpu(), values.double().cpu()
    d_in, n = k.shape
    lam = ridge * stats.whitened_dim()
    ck = stats.inverse(k)                                  # Σ⁻¹K, d_in x n
    gram = k.T @ ck
    g, e = torch.linalg.eigh((gram + gram.T) / 2)
    keep = g > g.max().clamp_min(1e-300) * 1e-10           # drop directions of duplicated keys
    g, e = g[keep], e[:, keep]
    if g.numel() == 0:
        zero_a, zero_b = torch.zeros(v.shape[0], 1), torch.zeros(1, d_in)
        return zero_a, zero_b, {"keys": n, "rank": 0, "fit": 0.0, "kept_fraction": 0.0}
    p, s, rt = torch.linalg.svd((v @ e) * torch.sqrt(g / (g + lam)), full_matrices=False)
    r = min(rank, s.numel())
    a = p[:, :r] * s[:r]
    b = ((rt[:r] / torch.sqrt(g * (g + lam))) @ e.T) @ ck.T
    total = float(v.square().sum())
    residual = float((a @ (b @ k) - v).square().sum())
    return a.float(), b.float(), {
        "keys": n,
        "rank": r,
        "fit": 1.0 - residual / total if total > 0 else 0.0,
        "kept_fraction": float(s[:r].square().sum() / s.square().sum().clamp_min(1e-300)),
        "mean_whitened_norm": float(g.mean()),
        "lambda": lam,
    }


def objective(a: torch.Tensor, b: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
              moment: torch.Tensor, ridge: float) -> float:
    """J(AB) with an explicit, unshrunk Σ (so E[zᵀ Σ⁻¹ z] = d_in), for checking the solver."""
    delta = a.double() @ b.double()
    lam = ridge * keys.shape[0]
    fit = (delta @ keys.double() - values.double()).square().sum()
    return float(fit + lam * torch.trace(delta @ moment.double() @ delta.T))
