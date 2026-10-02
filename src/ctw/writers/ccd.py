"""CCD: closed-form context distillation.

1. Self-queries (ctw.queries): the frozen model asks itself questions about the document, and the
   teacher (the same model with the document in the prompt) answers them.
2. Targets: at the top memory layer, an offset δ on every shared position of each query. δ makes
   the student, which reads only the query, match the teacher's answer distribution, and the
   anchor term keeps it near the teacher's own residual stream:
       min_δ  mean_t KL(p_teacher ‖ p_student(δ)) + anchor · mean_t ‖δ_t − r_t‖² / ‖h_t‖²
   with r_t the teacher-minus-student residual at the top layer.
3. Solve: from the lowest memory layer up, each layer takes an equal share of the offset still
   missing at the top layer, fitted in closed form by ctw.solve.covariance_ridge. Head positions
   (the template before the question, identical in every prompt) get the target 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..memory import LowRankDelta, MemoryState, as_tokens
from ..queries import Query, synthesize
from ..solve import covariance_ridge
from .base import Context, Writer, register


@dataclass
class CCDParams:
    rank: int = 32            # rank of ΔW per layer
    ridge: float = 0.1        # a typical generic key keeps 1/(1+ridge) of its target
    steps: int = 30           # Adam steps on δ; 0 keeps δ = r, the teacher's residual
    lr: float = 0.05          # step size in units of ‖h_t‖/√d per coordinate
    anchor: float = 0.5       # weight on staying near the teacher's residual
    questions: int = 16       # questions the model writes about the document
    cloze: bool = True        # add one cloze query per sentence
    answer_tokens: int = 12   # longest teacher answer


def _add_tail(output: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Add tail (m x d) to the last m positions of a (1, n, d) or (n, d) output."""
    flat = as_tokens(output)
    pad = torch.zeros(flat.shape[1] - tail.shape[0], tail.shape[1], device=tail.device, dtype=tail.dtype)
    return output + torch.cat([pad, tail]).reshape(output.shape).to(output.dtype)


@register("ccd")
class CCDWriter(Writer):
    Params = CCDParams

    def write(self, ctx: Context, document: str) -> MemoryState:
        if ctx.stats is None:
            raise ValueError("ccd needs key statistics (ctx.stats)")
        ctx.stats.ensure(ctx.layers)
        queries = synthesize(ctx, document, self.p.questions, self.p.cloze, self.p.answer_tokens)
        top = ctx.layers[-1]
        base, targets, kl = [], [], []
        for q in queries:
            h0, delta, kls = self._target(ctx, q, top)
            base.append(h0)
            targets.append(delta)
            kl.append(kls)
        heads = {}
        for q in queries:
            if q.head:
                ids = q.student[:, :q.head]
                heads.setdefault(tuple(ids[0].tolist()), ids)
        head_base = [(ids, self._read(ctx, ids, top, top, None)[1]) for ids in heads.values()]

        state = MemoryState({})
        layers = {}
        try:
            for i, layer in enumerate(ctx.layers):
                share = len(ctx.layers) - i
                keys, values = [], []
                for q, h0, delta in zip(queries, base, targets):
                    z, h = self._read(ctx, q.student, layer, top, state)
                    keys.append(z[q.student_shared])
                    values.append((h0 + delta - h[q.student_shared]) / share)
                for ids, h0 in head_base:
                    z, h = self._read(ctx, ids, layer, top, state)
                    keys.append(z)
                    values.append((h0 - h) / share)
                stats = ctx.stats.get(layer)
                a, b, report = covariance_ridge(torch.cat(keys).T, torch.cat(values).T, stats,
                                                self.p.ridge, self.p.rank)
                delta_w = LowRankDelta(a.to(ctx.device), b.to(ctx.device))
                state.deltas[layer] = delta_w
                report["disturbance"] = stats.energy(delta_w) / ctx.stats.weight_energy(layer)
                layers[layer] = report
        finally:
            ctx.hooks.clear()
        state.meta = {
            "writer": self.name,
            "queries": len(queries),
            "generated": sum(q.source == "generated" for q in queries),
            "cloze": sum(q.source == "cloze" for q in queries),
            # Mean answer KL to the teacher at the top layer: no offset, δ = r, and the fitted δ.
            "kl_off": sum(k[0] for k in kl) / len(kl),
            "kl_residual": sum(k[1] for k in kl) / len(kl),
            "kl_target": sum(k[2] for k in kl) / len(kl),
            "layers": layers,
        }
        return state

    @torch.no_grad()
    def _read(self, ctx: Context, ids: torch.Tensor, layer: int, top: int,
              state: MemoryState | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Out-projection input at `layer` and output of decoder layer `top`, per position (CPU)."""
        got = {}
        handles = [
            ctx.adapter.out_proj(layer).register_forward_pre_hook(
                lambda _m, inp: got.__setitem__("z", as_tokens(inp[0])[0].float().cpu())),
            ctx.adapter.layers[top].register_forward_hook(
                lambda _m, _i, out: got.__setitem__("h", as_tokens(ctx.adapter.layer_output(out))[0]
                                                    .float().cpu())),
        ]
        ctx.hooks.set(state)
        try:
            ctx.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
            ctx.hooks.set(None)
        return got["z"], got["h"]

    def _target(self, ctx: Context, q: Query, top: int):
        """Student residual h0 and fitted offset δ at the top layer on the shared positions,
        with the answer KL at no offset, at δ = r, and at the fitted δ."""
        ctx.hooks.set(None)
        got = {}

        def keep(_m, _i, out):
            got["h"] = as_tokens(ctx.adapter.layer_output(out))[0].float()

        with torch.no_grad():
            handle = ctx.adapter.layers[top].register_forward_hook(keep)
            try:
                t_logits = ctx.model(input_ids=q.teacher, use_cache=False).logits
                t_h = got["h"][q.teacher_shared]
                ctx.model(input_ids=q.student, use_cache=False)
                s_h = got["h"][q.student_shared]
            finally:
                handle.remove()
            t_logp = F.log_softmax(t_logits[0, q.teacher_answer].float(), dim=-1)
        residual = t_h - s_h
        scale = s_h.norm(dim=-1, keepdim=True) / math.sqrt(s_h.shape[-1])
        u = torch.zeros_like(residual, requires_grad=True)
        opt = torch.optim.Adam([u], lr=self.p.lr)
        offset = {"tail": residual}
        handle = ctx.adapter.out_proj(top).register_forward_hook(
            lambda _m, _i, out: _add_tail(out, offset["tail"]))

        def kl() -> torch.Tensor:
            logits = ctx.model(input_ids=q.student, use_cache=False).logits[0, q.student_answer].float()
            return F.kl_div(F.log_softmax(logits, dim=-1), t_logp, log_target=True, reduction="batchmean")

        try:
            with torch.no_grad():
                offset["tail"] = torch.zeros_like(residual)
                off = float(kl())
                offset["tail"] = residual
                start = float(kl())
            for _ in range(self.p.steps):
                offset["tail"] = residual + scale * u
                loss = kl() + self.p.anchor * (u.square().sum(-1) / u.shape[-1]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            offset["tail"] = (residual + scale * u).detach()
            with torch.no_grad():
                end = float(kl())
        finally:
            handle.remove()
        return s_h.cpu(), offset["tail"].cpu(), (off, start, end)
