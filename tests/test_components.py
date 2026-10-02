from __future__ import annotations

import pytest
import torch

from ctw.adapter import ModelAdapter, resolve_layers
from ctw.device import load_model
from ctw.memory import DenseDelta, LowRankDelta, MemoryHooks, MemoryState

from .conftest import ARCHS


def test_resolve_layers():
    assert resolve_layers(32, 36) == [32]
    assert resolve_layers(-1, 36) == [35]
    assert resolve_layers(0.89, 36) == [32]
    assert resolve_layers(0.89, 28) == [25]
    assert resolve_layers("last:3", 10) == [7, 8, 9]
    assert resolve_layers(["first:2", 5, "-1"], 10) == [0, 1, 5, 9]
    with pytest.raises(ValueError):
        resolve_layers(40, 36)
    with pytest.raises(ValueError):
        resolve_layers("middle", 36)


EXPECTED = {
    "qwen3": ("model.layers", "mlp.down_proj", (64, 32)),
    "llama": ("model.layers", "mlp.down_proj", (64, 32)),
    "gpt2": ("transformer.h", "mlp.c_proj", (128, 32)),
    "gpt_neox": ("gpt_neox.layers", "mlp.dense_4h_to_h", (64, 32)),
    "opt": ("model.decoder.layers", "fc2", (64, 32)),
}


@pytest.mark.parametrize("arch", ARCHS)
def test_adapter_detects_layout(arch, model_dirs):
    model, _ = load_model(model_dirs[arch], device="cpu", dtype="float32")
    ad = ModelAdapter.from_model(model)
    layers_path, out_proj, dims = EXPECTED[arch]
    assert (ad.layers_path, ad.out_proj_path, ad.dims(0)) == (layers_path, out_proj, dims)
    assert all(not p.requires_grad for p in model.parameters())


def test_adapter_manual_override_and_error(model_dirs):
    model, _ = load_model(model_dirs["qwen3"], device="cpu", dtype="float32")
    ad = ModelAdapter.from_model(model, layers_path="model.layers", out_proj="mlp.up_proj")
    assert ad.dims(0) == (32, 64)
    with pytest.raises(ValueError, match="layers_path"):
        ModelAdapter.from_model(model, layers_path="model.nothing")


@pytest.mark.parametrize("arch", ARCHS)
def test_hooks_identity_and_effect(arch, model_dirs):
    model, _ = load_model(model_dirs[arch], device="cpu", dtype="float32")
    ad = ModelAdapter.from_model(model)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    base = model(input_ids=ids).logits
    d_in, d_out = ad.dims(0)
    with MemoryHooks(ad, [0]) as hooks:
        hooks.set(MemoryState({0: DenseDelta(torch.zeros(d_out, d_in))}))
        assert torch.allclose(model(input_ids=ids).logits, base, atol=1e-6)
        hooks.set(MemoryState({0: LowRankDelta(torch.randn(d_out, 2), torch.randn(2, d_in))}))
        assert not torch.allclose(model(input_ids=ids).logits, base, atol=1e-4)
        hooks.clear()
        assert torch.allclose(model(input_ids=ids).logits, base, atol=1e-6)
    assert torch.allclose(model(input_ids=ids).logits, base, atol=1e-6)


def test_lowrank_matches_dense_and_gpt2_orientation(model_dirs):
    """A low-rank ΔW and its dense product give the same output, also through GPT-2 Conv1D."""
    model, _ = load_model(model_dirs["gpt2"], device="cpu", dtype="float32")
    ad = ModelAdapter.from_model(model)
    d_in, d_out = ad.dims(1)
    a, b = torch.randn(d_out, 3), torch.randn(3, d_in)
    ids = torch.tensor([[3, 4, 5]])
    with MemoryHooks(ad, [1]) as hooks:
        hooks.set(MemoryState({1: LowRankDelta(a, b)}))
        low = model(input_ids=ids).logits
        hooks.set(MemoryState({1: DenseDelta(a @ b)}))
        dense = model(input_ids=ids).logits
    assert torch.allclose(low, dense, atol=1e-4)


def test_state_save_load_random(tmp_path):
    s = MemoryState({3: LowRankDelta(torch.randn(8, 2), torch.randn(2, 16)), 5: DenseDelta(torch.randn(8, 16))},
                    {"writer": "x"})
    s.save(tmp_path / "s.safetensors")
    t = MemoryState.load(tmp_path / "s.safetensors")
    assert t.meta == {"writer": "x"} and set(t.deltas) == {3, 5}
    assert torch.equal(t.deltas[3].a, s.deltas[3].a) and torch.equal(t.deltas[5].w, s.deltas[5].w)
    r = s.random_like(1)
    for i in s.deltas:
        assert abs(r.deltas[i].frobenius() - s.deltas[i].frobenius()) < 1e-3 * s.deltas[i].frobenius()
    assert r.deltas[3].rank == 2
    assert s.nbytes(2) == 2 * (8 * 2 + 2 * 16 + 8 * 16)
