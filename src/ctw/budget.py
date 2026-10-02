"""Equal-byte comparison between a weight state and the KV cache."""

from __future__ import annotations

from .adapter import ModelAdapter


def kv_params_per_token(config) -> int | None:
    """2 (K and V) x layers x KV heads x head dim, read from a Hugging Face config."""
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        return None
    return 2 * layers * kv_heads * head_dim


def dense_state_params(adapter: ModelAdapter, layers: list[int]) -> int:
    total = 0
    for i in layers:
        d_in, d_out = adapter.dims(i)
        total += d_in * d_out
    return total


def lowrank_state_params(adapter: ModelAdapter, layers: list[int], rank: int) -> int:
    total = 0
    for i in layers:
        d_in, d_out = adapter.dims(i)
        total += rank * (d_in + d_out)
    return total


def kv_equivalent_tokens(adapter: ModelAdapter, state_params: int) -> float | None:
    per_token = kv_params_per_token(adapter.model.config)
    return state_params / per_token if per_token else None
