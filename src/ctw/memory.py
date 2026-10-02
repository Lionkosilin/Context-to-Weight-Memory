"""Memory state: per-layer ΔW on the MLP output projection, applied through forward hooks.

The backbone weights never change. A hook adds ΔW·z to the projection output, where z is the
projection input. This works for any weight layout (nn.Linear, Conv1D) because ΔW is stored in
(d_out x d_in) orientation and applied directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .adapter import ModelAdapter


@dataclass
class LowRankDelta:
    a: torch.Tensor  # d_out x r
    b: torch.Tensor  # r x d_in

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.b.float()), self.a.float())

    def frobenius(self) -> float:
        a, b = self.a.detach().float(), self.b.detach().float()
        return float(torch.sum((a.T @ a) * (b @ b.T)).clamp_min(0.0).sqrt())

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"a": self.a, "b": self.b}

    @property
    def rank(self) -> int:
        return int(self.a.shape[1])


@dataclass
class DenseDelta:
    w: torch.Tensor  # d_out x d_in

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w.float())

    def frobenius(self) -> float:
        return float(self.w.detach().float().norm())

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"w": self.w}

    @property
    def rank(self) -> int | None:
        return None


Delta = LowRankDelta | DenseDelta


@dataclass
class MemoryState:
    """One document's memory: layer index -> ΔW."""

    deltas: dict[int, Delta]
    meta: dict = field(default_factory=dict)

    def frobenius(self) -> float:
        return sum(d.frobenius() ** 2 for d in self.deltas.values()) ** 0.5

    def nbytes(self, bytes_per_param: int = 2) -> int:
        return bytes_per_param * sum(t.numel() for d in self.deltas.values() for t in d.tensors().values())

    def to(self, device=None, dtype=None) -> MemoryState:
        def mv(t):
            return t.detach().to(device=device or t.device, dtype=dtype or t.dtype)
        out = {}
        for i, d in self.deltas.items():
            out[i] = LowRankDelta(mv(d.a), mv(d.b)) if isinstance(d, LowRankDelta) else DenseDelta(mv(d.w))
        return MemoryState(out, dict(self.meta))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"{i}.{k}": t.detach().contiguous().cpu()
                   for i, d in self.deltas.items() for k, t in d.tensors().items()}
        save_file(tensors, str(path), metadata={"meta": json.dumps(self.meta)})

    @classmethod
    def load(cls, path: str | Path) -> MemoryState:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("meta", "{}"))
        flat = load_file(str(path))
        grouped: dict[int, dict[str, torch.Tensor]] = {}
        for key, t in flat.items():
            i, _, k = key.partition(".")
            grouped.setdefault(int(i), {})[k] = t
        deltas = {i: LowRankDelta(g["a"], g["b"]) if "a" in g else DenseDelta(g["w"])
                  for i, g in sorted(grouped.items())}
        return cls(deltas, meta)

    def random_like(self, seed: int) -> MemoryState:
        """Random state with the same layers, shapes, rank, and per-layer Frobenius norm."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out: dict[int, Delta] = {}
        for i, d in self.deltas.items():
            target = d.frobenius()
            if isinstance(d, LowRankDelta):
                a = torch.randn(d.a.shape, generator=gen)
                b = torch.randn(d.b.shape, generator=gen)
                r = LowRankDelta(a, b)
                norm = r.frobenius()
                if norm > 0:
                    a.mul_(target / norm)
                out[i] = LowRankDelta(a.to(d.a), b.to(d.b))
            else:
                w = torch.randn(d.w.shape, generator=gen)
                w.mul_(target / max(float(w.norm()), 1e-12))
                out[i] = DenseDelta(w.to(d.w))
        return MemoryState(out, {**self.meta, "control": "random_same_shape_norm", "seed": seed})


def as_tokens(x: torch.Tensor) -> torch.Tensor:
    """Projection input as (batch, seq, d). OPT flattens to (seq, d) before its MLP; runs use batch 1."""
    return x if x.dim() == 3 else x.unsqueeze(0)


class MemoryHooks:
    """Attach a MemoryState to a model. One instance per model; swap states with set()/clear()."""

    def __init__(self, adapter: ModelAdapter, layers: list[int]):
        self.adapter = adapter
        self.layers = list(layers)
        self.state: MemoryState | None = None
        self.scale = 1.0
        self.last_input: dict[int, torch.Tensor] = {}
        self._handles = []
        for idx in self.layers:
            self._handles.append(adapter.out_proj(idx).register_forward_hook(self._hook(idx)))

    def _hook(self, idx: int):
        def hook(_module, inputs, output):
            x = inputs[0]
            self.last_input[idx] = as_tokens(x)[:, -1].detach().float()
            if self.state is None or idx not in self.state.deltas or self.scale == 0.0:
                return output
            delta = self.state.deltas[idx].apply(x.float())
            return output + (self.scale * delta).to(output.dtype)
        return hook

    def set(self, state: MemoryState | None, scale: float = 1.0) -> None:
        if state is not None:
            missing = set(state.deltas) - set(self.layers)
            if missing:
                raise ValueError(f"state has layers {sorted(missing)} with no hook attached")
        self.state, self.scale = state, float(scale)

    def clear(self) -> None:
        self.state = None
        self.last_input.clear()

    def close(self) -> None:
        self.clear()
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> MemoryHooks:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@torch.no_grad()
def capture(adapter: ModelAdapter, input_ids: torch.Tensor, layers: list[int]
            ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """One forward pass. Returns layer -> (projection input, projection output), CPU float32."""
    got: dict[int, list] = {i: [None, None] for i in layers}
    handles = []
    for idx in layers:
        mod = adapter.out_proj(idx)

        def pre(_m, inputs, i=idx):
            got[i][0] = as_tokens(inputs[0]).detach().float().cpu()

        def post(_m, _inputs, output, i=idx):
            got[i][1] = as_tokens(output).detach().float().cpu()

        handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    try:
        adapter.model(input_ids=input_ids.to(adapter.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {i: (x, y) for i, (x, y) in got.items()}
