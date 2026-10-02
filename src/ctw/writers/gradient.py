"""Inference-time gradient writing: ΔW is trained on the document while the backbone stays frozen.

Each chunk of the document gives one optimizer step. ΔW is dense or low-rank on every memory layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..memory import DenseDelta, LowRankDelta, MemoryState
from ..prompting import encode
from .base import Context, Writer, register


@dataclass
class GradientParams:
    lr: float = 1e-4
    optimizer: str = "sgd"        # sgd | adam
    passes: int = 8
    chunk: int = 512
    kind: str = "dense"           # dense | lowrank
    rank: int = 8                 # lowrank only
    init_std: float = 0.01        # lowrank only: std of B; A starts at zero
    max_norm: float | None = None  # cap on ||ΔW||_F after each step


def _trainable(ctx: Context, p: GradientParams) -> tuple[MemoryState, list[torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(ctx.seed)
    deltas, params = {}, []
    for i in ctx.layers:
        d_in, d_out = ctx.adapter.dims(i)
        if p.kind == "dense":
            w = torch.zeros(d_out, d_in, device=ctx.device, requires_grad=True)
            deltas[i], params = DenseDelta(w), params + [w]
        elif p.kind == "lowrank":
            a = torch.zeros(d_out, p.rank, device=ctx.device, requires_grad=True)
            b = (torch.randn(p.rank, d_in, generator=gen) * p.init_std).to(ctx.device).requires_grad_(True)
            deltas[i], params = LowRankDelta(a, b), params + [a, b]
        else:
            raise ValueError(f"kind must be dense or lowrank, not {p.kind!r}")
    return MemoryState(deltas), params


def _optimizer(p: GradientParams, params):
    if p.optimizer == "sgd":
        return torch.optim.SGD(params, lr=p.lr)
    if p.optimizer == "adam":
        return torch.optim.Adam(params, lr=p.lr)
    raise ValueError(f"optimizer must be sgd or adam, not {p.optimizer!r}")


@torch.no_grad()
def _cap(state: MemoryState, max_norm: float | None) -> None:
    if max_norm is None:
        return
    norm = state.frobenius()
    if norm > max_norm:
        for d in state.deltas.values():
            (d.w if isinstance(d, DenseDelta) else d.a).mul_(max_norm / norm)


def _chunks(n: int, chunk: int, start: int = 0) -> list[tuple[int, int]]:
    return [(s, min(n, s + chunk)) for s in range(start, n, chunk)]


class GradientWriter(Writer):
    Params = GradientParams

    def write(self, ctx: Context, document: str) -> MemoryState:
        ids = encode(ctx.tok, document, ctx.device)
        state, params = _trainable(ctx, self.p)
        opt = _optimizer(self.p, params)
        steps = 0
        try:
            for _ in range(self.p.passes):
                for start, end in self.spans(ids.shape[1]):
                    ctx.hooks.set(state)
                    loss = self.loss(ctx, ids, start, end, state)
                    if loss is None:
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    _cap(state, self.p.max_norm)
                    steps += 1
        finally:
            ctx.hooks.clear()
        out = state.to()
        out.meta = {"writer": self.name, "steps": steps}
        return out

    def spans(self, n: int) -> list[tuple[int, int]]:
        return _chunks(n, self.p.chunk)

    def loss(self, ctx, ids, start, end, state):
        raise NotImplementedError


@register("ntp")
class NTPWriter(GradientWriter):
    """Next-token prediction on each chunk."""

    def loss(self, ctx, ids, start, end, state):
        chunk = ids[:, start:end]
        if chunk.shape[1] < 2:
            return None
        return ctx.model(input_ids=chunk, labels=chunk, use_cache=False).loss


@dataclass
class TTCDParams(GradientParams):
    prefix: int = 512   # tokens every teacher sees
    recent: int = 2048  # earlier tokens the teacher also sees


@register("ttcd")
class TTCDWriter(GradientWriter):
    """Teacher sees prefix + recent text + chunk; the student sees the chunk. MSE on final hidden states."""

    Params = TTCDParams

    def spans(self, n):
        return _chunks(n, self.p.chunk, start=self.p.prefix)

    def loss(self, ctx, ids, start, end, state):
        student = ids[:, start:end]
        teacher = torch.cat([ids[:, :self.p.prefix], ids[:, max(self.p.prefix, end - self.p.recent):end]], dim=1)
        if teacher.shape[1] == student.shape[1]:
            return None
        n = student.shape[1]
        with torch.no_grad():
            t = ctx.model(input_ids=teacher, output_hidden_states=True, use_cache=False)
            target = t.hidden_states[-1][:, -n:].float()
        s = ctx.model(input_ids=student, output_hidden_states=True, use_cache=False)
        return F.mse_loss(s.hidden_states[-1].float(), target)


@dataclass
class DCDParams(GradientParams):
    lr: float = 1e-4
    optimizer: str = "adam"
    recent: int = 2048         # earlier tokens the teacher sees
    hidden_weight: float = 1.0  # λ on the per-layer hidden-state L1 term


@register("dcd")
class DCDWriter(GradientWriter):
    """Deep context distillation: the frozen backbone without ΔW reads earlier text + chunk;
    the student with ΔW reads the chunk. KL on logits plus relative L1 on every hidden state."""

    Params = DCDParams

    def loss(self, ctx, ids, start, end, state):
        student = ids[:, start:end]
        teacher = ids[:, max(0, end - self.p.recent - self.p.chunk):end]
        m = student.shape[1]
        with torch.no_grad():
            ctx.hooks.set(None)
            t = ctx.model(input_ids=teacher, output_hidden_states=True, use_cache=False)
            t_logp = F.log_softmax(t.logits[:, -m:].float(), dim=-1)
            t_h = [h[:, -m:].float() for h in t.hidden_states[1:]]
            ctx.hooks.set(state)
        s = ctx.model(input_ids=student, output_hidden_states=True, use_cache=False)
        s_logp = F.log_softmax(s.logits.float(), dim=-1)
        kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1).mean()
        l1 = sum((sh.float() - th).abs().sum() / th.abs().sum().clamp_min(1e-6)
                 for sh, th in zip(s.hidden_states[1:], t_h)) / len(t_h)
        return kl + self.p.hidden_weight * l1
