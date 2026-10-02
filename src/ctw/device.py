"""Device and dtype selection, and model loading."""

from __future__ import annotations

import torch

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        if name not in DTYPES:
            raise ValueError(f"unknown dtype {name!r}; choose from {sorted(DTYPES)}")
        return DTYPES[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_model(model_id: str, device: str = "auto", dtype: str = "auto",
               trust_remote_code: bool = False):
    """Load a frozen causal LM and its tokenizer. Every parameter has requires_grad=False."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(device)
    dt = resolve_dtype(dtype, dev)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dt, trust_remote_code=trust_remote_code
    ).to(dev).eval()
    freeze(model)
    return model, tok


def freeze(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
