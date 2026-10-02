"""Key statistics: the second moment Σ = E[z zᵀ] of the out-projection input on generic text.

Σ is the metric of the closed-form write: tr(ΔW Σ ΔWᵀ) is the expected squared change ΔW makes to
the layer output on generic text. Each layer's eigendecomposition is computed once and cached.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .memory import DenseDelta, LowRankDelta, as_tokens
from .prompting import encode


@dataclass
class KeyStats:
    eigvecs: torch.Tensor   # d x d, float32; columns are eigenvectors of E[z zᵀ]
    eigvals: torch.Tensor   # d, float64; raw eigenvalues
    tokens: int
    shrink: float           # weight on the isotropic part: (1 - s) Λ + s · mean(Λ)

    def __post_init__(self):
        lam = self.eigvals.double().clamp_min(0.0)
        self.spectrum = (1.0 - self.shrink) * lam + self.shrink * lam.mean()
        if float(self.spectrum.min()) <= 0.0:
            raise ValueError("key covariance is singular; raise stats.shrink above 0")

    @classmethod
    def from_moment(cls, moment: torch.Tensor, tokens: int, shrink: float) -> KeyStats:
        vals, vecs = torch.linalg.eigh(moment.double())
        return cls(vecs.float(), vals, tokens, shrink)

    @property
    def dim(self) -> int:
        return int(self.eigvecs.shape[0])

    def whitened_dim(self) -> float:
        """E[zᵀ Σ⁻¹ z] over generic keys: d without shrinkage, less with it."""
        raw = self.eigvals.double().clamp_min(0.0)
        return float((raw / self.spectrum).sum())

    def effective_dim(self) -> float:
        """(tr Σ)² / tr(Σ²): how many directions the keys really spread over."""
        s = self.spectrum
        return float(s.sum() ** 2 / (s * s).sum())

    def inverse(self, k: torch.Tensor) -> torch.Tensor:
        """Σ⁻¹ k for k of shape (d, n), float64."""
        u = self.eigvecs
        proj = (u.T @ k.to(u)).double() / self.spectrum[:, None]
        return (u @ proj.float()).double()

    def energy(self, delta: LowRankDelta | DenseDelta) -> float:
        """tr(ΔW Σ ΔWᵀ) = E‖ΔW z‖² over generic keys."""
        u, root = self.eigvecs, self.spectrum.sqrt().float()
        if isinstance(delta, LowRankDelta):
            m = (delta.b.detach().float().cpu() @ u) * root
            return float(torch.sum((delta.a.detach().float().cpu().T @ delta.a.detach().float().cpu())
                                   * (m @ m.T)))
        m = (delta.w.detach().float().cpu() @ u) * root
        return float(m.square().sum())

    def weight_energy(self, weight: torch.Tensor) -> float:
        """tr(W Σ Wᵀ): the energy of the layer's own output on generic text."""
        return self.energy(DenseDelta(weight))


class StatsCache:
    """Loads or computes KeyStats per layer from the `stats` config section."""

    def __init__(self, model, tok, adapter, hooks, cfg: dict, model_id: str):
        self.model, self.tok, self.adapter, self.hooks, self.cfg = model, tok, adapter, hooks, cfg
        self.model_id = model_id
        self._stats: dict[int, KeyStats] = {}
        self._weight_energy: dict[int, float] = {}

    def get(self, layer: int) -> KeyStats:
        self.ensure([layer])
        return self._stats[layer]

    def weight_energy(self, layer: int) -> float:
        if layer not in self._weight_energy:
            self._weight_energy[layer] = self.get(layer).weight_energy(self.adapter.weight(layer))
        return self._weight_energy[layer]

    def ensure(self, layers: list[int]) -> None:
        missing = []
        for layer in layers:
            if layer in self._stats:
                continue
            path = self._path(layer)
            if path.exists():
                blob = torch.load(path, map_location="cpu")
                self._stats[layer] = KeyStats(blob["eigvecs"], blob["eigvals"], blob["tokens"],
                                              self.cfg["shrink"])
            else:
                missing.append(layer)
        if missing:
            self._compute(missing)

    # ------------------------------------------------------------ computation

    def _tag(self) -> str:
        c = self.cfg
        if c["source"] == "sample":
            return f"sample-{c['tokens']}-{c['chunk']}"
        if c["source"] == "text":
            h = hashlib.sha1()
            for p in c["text"]:
                h.update(Path(p).read_bytes())
            return f"text-{h.hexdigest()[:12]}-{c['tokens']}-{c['chunk']}"
        raise ValueError(f"stats.source must be sample or text, not {c['source']!r}")

    def _path(self, layer: int) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "--", self.model_id.strip("/"))
        return Path(self.cfg["cache"].format(model=name)) / f"layer{layer}-{self._tag()}.pt"

    def _sequences(self):
        c, tok, model = self.cfg, self.tok, self.model
        device = self.adapter.device
        if c["source"] == "text":
            stream = torch.cat([encode(tok, Path(p).read_text(encoding="utf-8"))[0] for p in c["text"]])
            for s in range(0, min(stream.numel(), c["tokens"] + 1), c["chunk"]):
                piece = stream[s:s + c["chunk"]]
                if piece.numel() > 1:
                    yield piece[None].to(device)
            return
        start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else start
        made = 0
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)
            while made < c["tokens"]:
                ids = torch.full((c["batch"], 1), start, dtype=torch.long, device=device)
                out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                     temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=c["chunk"],
                                     min_new_tokens=c["chunk"], pad_token_id=pad)
                for row in out:
                    yield row[None]
                    made += row.numel() - 1

    @torch.no_grad()
    def _compute(self, layers: list[int]) -> None:
        device = self.adapter.device
        acc = {i: torch.zeros(self.adapter.dims(i)[0], self.adapter.dims(i)[0], device=device)
               for i in layers}
        count = 0

        def pre(i):
            def hook(_m, inputs):
                # Position 0 is the attention sink; its keys are outliers no read position shares.
                z = as_tokens(inputs[0])[0, 1:].float()
                acc[i].addmm_(z.T, z)
            return hook

        handles = [self.adapter.out_proj(i).register_forward_pre_hook(pre(i)) for i in layers]
        state, scale = self.hooks.state, self.hooks.scale
        self.hooks.set(None)
        try:
            for ids in self._sequences():
                self.model(input_ids=ids, use_cache=False)
                count += ids.shape[1] - 1
                if count >= self.cfg["tokens"]:
                    break
        finally:
            for h in handles:
                h.remove()
            self.hooks.set(state, scale)
        if count == 0:
            raise ValueError("no tokens for key statistics; check stats.source and stats.text")
        eig_device = device if device.type == "cuda" else torch.device("cpu")
        for i in layers:
            moment = (acc[i].to(eig_device).double() / count)
            vals, vecs = torch.linalg.eigh(moment)
            blob = {"eigvecs": vecs.float().cpu(), "eigvals": vals.cpu(), "tokens": count,
                    "model": self.model_id, "layer": i, "source": self._tag()}
            path = self._path(i)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, path)
            self._stats[i] = KeyStats(blob["eigvecs"], blob["eigvals"], count, self.cfg["shrink"])
            del acc[i]
