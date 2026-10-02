"""Analytic residual writing: one closed-form ridge solve per layer, no training."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..memory import LowRankDelta, MemoryState, capture
from ..prompting import encode
from .base import Context, Writer, register

# Fixed before any document exists. They name no entity, value, question, or answer.
UNIVERSAL_PROBES = (
    "Read the preceding passage carefully and retain its exact details for later use.",
    "Represent the concrete names, phrases, codes, quantities, locations, and dates in the preceding passage.",
    "Preserve every factual relation in the preceding passage so that differently worded questions can be answered later.",
    "Form a compact internal record of the preceding passage without adding unsupported information.",
)


@dataclass
class AnalyticParams:
    rank: int = 16                 # probe positions kept per layer = max rank of ΔW
    ridge: float = 1e-2            # λ, relative to the mean diagonal of XᵀX
    probes: list[str] = field(default_factory=lambda: list(UNIVERSAL_PROBES))


@register("analytic")
class AnalyticWriter(Writer):
    """Teacher reads [document; probe], student reads the probe alone.
    ΔW = R (XᵀX + λI)⁻¹ Xᵀ over the r positions with the largest teacher-student residual."""

    Params = AnalyticParams

    def write(self, ctx: Context, document: str) -> MemoryState:
        samples: dict[int, list[tuple[torch.Tensor, torch.Tensor, float]]] = {i: [] for i in ctx.layers}
        doc = encode(ctx.tok, document, ctx.device)
        ctx.hooks.set(None)
        for probe in self.p.probes:
            suffix = encode(ctx.tok, "\n\n" + probe, ctx.device)
            n = suffix.shape[1]
            teacher = capture(ctx.adapter, torch.cat([doc, suffix], dim=1), ctx.layers)
            student = capture(ctx.adapter, suffix, ctx.layers)
            for i in ctx.layers:
                x = student[i][0][0, -n:]
                r = teacher[i][1][0, -n:] - student[i][1][0, -n:]
                samples[i] += [(x[t], r[t], float(r[t].norm())) for t in range(n)]
        deltas, fits = {}, {}
        for i in ctx.layers:
            top = sorted(samples[i], key=lambda s: s[2], reverse=True)[: self.p.rank]
            x = torch.stack([s[0] for s in top], dim=1)      # d_in x r
            res = torch.stack([s[1] for s in top], dim=1)    # d_out x r
            gram = x.T @ x
            lam = self.p.ridge * max(float(torch.diagonal(gram).mean()), 1e-8)
            a = res @ torch.linalg.solve(gram + lam * torch.eye(gram.shape[0]), torch.eye(gram.shape[0]))
            b = x.T.contiguous()
            pred = a @ (b @ x)
            before = float((res * res).sum())
            fits[i] = 0.0 if before <= 0 else 1.0 - float(((res - pred) ** 2).sum()) / before
            deltas[i] = LowRankDelta(a.contiguous(), b)
        return MemoryState(deltas, {"writer": self.name, "fit_fraction": fits})
