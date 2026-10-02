"""Inference-time gradient writing: ΔW is trained per document while the backbone stays frozen.

A writer prepares training items from the document, then takes one optimizer step per item for a
number of passes. ΔW is dense or low-rank on every memory layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..memory import DenseDelta, LowRankDelta, MemoryState
from ..prompting import encode
from ..queries import Query, synthesize
from .base import Context, Writer, register


@dataclass
class GradientParams:
    lr: float = 1e-4
    optimizer: str = "sgd"        # sgd | adam
    passes: int = 8
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


class GradientWriter(Writer):
    Params = GradientParams

    def write(self, ctx: Context, document: str) -> MemoryState:
        items = self.prepare(ctx, document)
        state, params = _trainable(ctx, self.p)
        opt = _optimizer(self.p, params)
        steps = 0
        try:
            for _ in range(self.p.passes):
                for item in items:
                    ctx.hooks.set(state)
                    loss = self.loss(ctx, item)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    _cap(state, self.p.max_norm)
                    steps += 1
        finally:
            ctx.hooks.clear()
        out = state.to()
        out.meta = {"writer": self.name, "steps": steps, "items": len(items)}
        return out

    def prepare(self, ctx: Context, document: str) -> list:
        raise NotImplementedError

    def loss(self, ctx: Context, item) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class NTPParams(GradientParams):
    chunk: int = 512


@register("ntp")
class NTPWriter(GradientWriter):
    """Next-token prediction on the document text, one step per chunk."""

    Params = NTPParams

    def prepare(self, ctx, document):
        ids = encode(ctx.tok, document, ctx.device)
        chunks = [ids[:, s:s + self.p.chunk] for s in range(0, ids.shape[1], self.p.chunk)]
        return [c for c in chunks if c.shape[1] >= 2]

    def loss(self, ctx, chunk):
        return ctx.model(input_ids=chunk, labels=chunk, use_cache=False).loss


@dataclass
class DCDParams(GradientParams):
    lr: float = 1e-3
    optimizer: str = "adam"
    passes: int = 4
    kind: str = "lowrank"
    rank: int = 16
    hidden_weight: float = 1.0  # λ on the per-layer hidden-state L1 term
    questions: int = 16         # questions the model writes about the document
    cloze: bool = True          # add one cloze query per sentence
    answer_tokens: int = 12     # longest teacher answer


@dataclass
class Target:
    query: Query
    logp: torch.Tensor           # answer positions x vocab: the teacher's log-probabilities
    hidden: list[torch.Tensor]   # per layer ΔW can change: shared positions x d_model


@register("dcd")
class DCDWriter(GradientWriter):
    """Deep context distillation over self-queries. The teacher reads the document and the query;
    the student with ΔW reads the query. KL on the answer tokens plus relative L1 on the output of
    every layer from the first memory layer up, over the positions both sequences share."""

    Params = DCDParams

    @torch.no_grad()
    def prepare(self, ctx, document):
        queries = synthesize(ctx, document, self.p.questions, self.p.cloze, self.p.answer_tokens)
        first = min(ctx.layers) + 1
        out = []
        for q in queries:
            t = ctx.model(input_ids=q.teacher, output_hidden_states=True, use_cache=False)
            out.append(Target(q, F.log_softmax(t.logits[0, q.teacher_answer].float(), dim=-1),
                              [h[0, q.teacher_shared].float() for h in t.hidden_states[first:]]))
        return out

    def loss(self, ctx, item):
        q = item.query
        s = ctx.model(input_ids=q.student, output_hidden_states=True, use_cache=False)
        s_logp = F.log_softmax(s.logits[0, q.student_answer].float(), dim=-1)
        kl = F.kl_div(s_logp, item.logp, log_target=True, reduction="batchmean")
        student = s.hidden_states[min(ctx.layers) + 1:]
        l1 = sum((sh[0, q.student_shared].float() - th).abs().sum() / th.abs().sum().clamp_min(1e-6)
                 for sh, th in zip(student, item.hidden)) / len(item.hidden)
        return kl + self.p.hidden_weight * l1
