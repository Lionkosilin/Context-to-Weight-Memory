"""Bit budget: n layers of W_down vs KV tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    n_layers: int
    hidden: int
    intermediate: int
    n_kv_heads: int
    head_dim: int
    n_attn_layers: int
    w_down_params: int
    kv_per_token: int
    k_tokens: int

    @property
    def ok(self) -> bool:
        return self.k_tokens >= 1024


def compute(
    n_layers: int = 8,
    hidden: int = 2560,
    intermediate: int = 9728,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    n_attn_layers: int = 36,
) -> Budget:
    w_down = n_layers * hidden * intermediate
    kv = n_attn_layers * 2 * n_kv_heads * head_dim
    return Budget(
        n_layers=n_layers,
        hidden=hidden,
        intermediate=intermediate,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_attn_layers=n_attn_layers,
        w_down_params=w_down,
        kv_per_token=kv,
        k_tokens=w_down // kv,
    )
