"""ACWC: Associative Context Weight Compiler.

fit() learns one compiler on training documents. write() turns a new document into rank-m factors
with one frozen forward pass, m = number of sentences. Each sentence must end with a
single-token value followed by punctuation, e.g. "The archive key was amber."
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..adapter import get_path
from ..memory import LowRankDelta, MemoryState, as_tokens
from ..prompting import answer_ids
from ..tasks import Episode
from .base import Context, Writer, register


@dataclass
class ACWCParams:
    steps: int = 600
    eval_every: int = 100
    lr: float = 2e-3
    weight_decay: float = 1e-4
    route_weight: float = 0.2
    route_temperature: float = 20.0
    key_rank: int = 64        # rank of the learned key-metric residual U Vᵀ
    value_rank: int = 0       # 0 keeps the value map fixed, so unseen values transfer
    initial_gain: float = 96.0
    value_source: str = "embedding"  # embedding | hidden


@dataclass
class Source:
    keys: torch.Tensor    # m x d_in
    values: torch.Tensor  # m x d_out


class Compiler(nn.Module):
    def __init__(self, d_in: int, d_out: int, key_rank: int, value_rank: int, initial_gain: float):
        super().__init__()
        self.key_log_diagonal = nn.Parameter(torch.zeros(d_in))
        self.key_left = nn.Parameter(torch.zeros(d_in, key_rank))
        self.key_right = nn.Parameter(torch.empty(d_in, key_rank))
        nn.init.normal_(self.key_right, std=0.005)
        self.value_left = nn.Parameter(torch.zeros(d_out, value_rank))
        self.value_right = nn.Parameter(torch.empty(d_out, value_rank))
        nn.init.normal_(self.value_right, std=0.02)
        self.log_gain = nn.Parameter(torch.tensor(math.log(initial_gain)))

    def factors(self, src: Source, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = src.keys.to(device=device, dtype=torch.float32)
        values = src.values.to(device=device, dtype=torch.float32)
        weighted = keys * torch.exp(self.key_log_diagonal.clamp(-3.0, 3.0))
        k = weighted + (weighted @ self.key_right) @ self.key_left.T
        b = k / k.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        values = F.normalize(values, dim=-1)
        values = values + (values @ self.value_right) @ self.value_left.T
        gain = torch.exp(self.log_gain.clamp(math.log(1.0), math.log(1024.0)))
        return (gain * values).T.contiguous(), b.contiguous(), k


def split_sentences(tok, document: str, device) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Document ids after the tokenizer's start token, and each sentence's span."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.strip()) if s.strip()]
    if not sentences:
        raise ValueError("document contains no sentence")
    start = tok("", return_tensors="pt").input_ids
    parts, spans, cursor = [start], [], start.shape[1]
    for k, s in enumerate(sentences):
        ids = tok((" " if k else "") + s, add_special_tokens=False, return_tensors="pt").input_ids
        parts.append(ids)
        spans.append((cursor, cursor + ids.shape[1]))
        cursor += ids.shape[1]
    return torch.cat(parts, dim=1).to(device), spans


@register("acwc")
class ACWCWriter(Writer):
    Params = ACWCParams
    trainable = True

    def __init__(self, params: dict):
        super().__init__(params)
        self.compiler: Compiler | None = None

    # ------------------------------------------------------------ write side

    def _layer(self, ctx: Context) -> int:
        if len(ctx.layers) != 1:
            raise ValueError(f"acwc writes one layer; memory.layers resolved to {ctx.layers}")
        return ctx.layers[0]

    @torch.no_grad()
    def source(self, ctx: Context, document: str) -> Source:
        """One forward over the raw document. No question or answer reaches this function."""
        layer = self._layer(ctx)
        out_proj = ctx.adapter.out_proj(layer)
        got: dict[str, torch.Tensor] = {}
        handles = [out_proj.register_forward_pre_hook(
            lambda _m, inp: got.__setitem__("z", as_tokens(inp[0]).detach().float().cpu()))]
        if self.p.value_source == "hidden":
            mlp_path = ctx.adapter.out_proj_path.rpartition(".")[0]
            if not mlp_path:
                raise ValueError("value_source=hidden needs an MLP module above out_proj")
            mlp = get_path(ctx.adapter.layers[layer], mlp_path)
            handles.append(mlp.register_forward_pre_hook(
                lambda _m, inp: got.__setitem__("h", as_tokens(inp[0]).detach().float().cpu())))
        ids, spans = split_sentences(ctx.tok, document, ctx.device)
        ctx.hooks.set(None)
        try:
            ctx.model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        emb = ctx.adapter.output_embeddings()
        keys, values = [], []
        for start, end in spans:
            # Last token is punctuation, the one before it is the value; the key excludes both.
            if end - start < 4:
                raise ValueError("each sentence needs at least two key tokens, a value, and punctuation")
            keys.append(got["z"][0, start:end - 2].mean(dim=0))
            if self.p.value_source == "hidden":
                values.append(got["h"][0, end - 2])
            else:
                # The value row is the token the reader emits as the answer.
                piece = ctx.tok.decode([int(ids[0, end - 2])]).strip()
                emitted = answer_ids(ctx.tok, piece, ctx.chat)
                if len(emitted) != 1:
                    raise ValueError(f"sentence value {piece!r} is not a single answer token")
                values.append(emb[emitted[0]].detach().float().cpu())
        return Source(torch.stack(keys), torch.stack(values))

    def _state(self, ctx: Context, a, b) -> MemoryState:
        return MemoryState({self._layer(ctx): LowRankDelta(a, b)})

    def write(self, ctx: Context, document: str) -> MemoryState:
        if self.compiler is None:
            raise RuntimeError("acwc needs fit() or load() before write()")
        with torch.no_grad():
            a, b, _ = self.compiler.factors(self.source(ctx, document), ctx.device)
        state = self._state(ctx, a, b)
        state.meta = {"writer": self.name, "rank": int(a.shape[1])}
        return state

    # ------------------------------------------------------------ fit side

    def fit(self, ctx: Context, train: list[Episode], dev: list[Episode]) -> dict:
        layer = self._layer(ctx)
        d_in, d_out = ctx.adapter.dims(layer)
        self.compiler = Compiler(d_in, d_out, self.p.key_rank, self.p.value_rank,
                                 self.p.initial_gain).to(ctx.device)
        train_src = [self.source(ctx, e.document) for e in train]
        dev_src = [self.source(ctx, e.document) for e in dev]
        examples = [
            (k, ctx.prompt.ids(ctx.tok, q.text, device=ctx.device),
             answer_ids(ctx.tok, q.gold, ctx.chat)[0], q.meta.get("slot"))
            for k, e in enumerate(train) for q in e.train_questions
        ]
        if not examples:
            raise ValueError("acwc fit needs episodes with train_questions")
        opt = torch.optim.AdamW(self.compiler.parameters(), lr=self.p.lr, weight_decay=self.p.weight_decay)
        rng = random.Random(ctx.seed + 41)
        best, best_state, marks = -float("inf"), copy.deepcopy(self.compiler.state_dict()), []
        self.compiler.train()
        for step in range(1, self.p.steps + 1):
            k, prompt, target, slot = examples[rng.randrange(len(examples))]
            a, b, keys = self.compiler.factors(train_src[k], ctx.device)
            ctx.hooks.set(self._state(ctx, a, b))
            logits = ctx.model(input_ids=prompt, use_cache=False).logits[:, -1].float()
            answer_loss = F.cross_entropy(logits, torch.tensor([target], device=ctx.device))
            route_loss = torch.zeros((), device=ctx.device)
            if slot is not None and self.p.route_weight:
                route = self.p.route_temperature * (
                    F.normalize(ctx.hooks.last_input[layer], dim=-1) @ F.normalize(keys, dim=-1).T)
                route_loss = F.cross_entropy(route, torch.tensor([slot], device=ctx.device))
            reg = (1e-5 * self.compiler.key_log_diagonal.square().mean()
                   + 1e-8 * self.compiler.key_left.square().mean()
                   + 1e-8 * self.compiler.key_right.square().mean())
            loss = answer_loss + self.p.route_weight * route_loss + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.compiler.parameters(), 1.0)
            opt.step()
            ctx.hooks.clear()
            if step % self.p.eval_every == 0 or step == self.p.steps:
                self.compiler.eval()
                score = self._dev_score(ctx, dev, dev_src)
                self.compiler.train()
                marks.append({"step": step, "answer_loss": answer_loss.item(),
                              "route_loss": route_loss.item(),
                              "gain": float(torch.exp(self.compiler.log_gain.detach())), **score})
                print(f"  step {step:4d} answer={marks[-1]['answer_loss']:.3f} "
                      f"route={marks[-1]['route_loss']:.3f} dev_logp={score['mean_gold_logp']:+.3f} "
                      f"dev_top1={score['first_token_accuracy']:.3f}", flush=True)
                if score["mean_gold_logp"] > best:
                    best, best_state = score["mean_gold_logp"], copy.deepcopy(self.compiler.state_dict())
        self.compiler.load_state_dict(best_state)
        self.compiler.eval()
        return {"marks": marks, "best_dev_mean_gold_logp": best,
                "compiler_parameters": sum(p.numel() for p in self.compiler.parameters())}

    @torch.no_grad()
    def _dev_score(self, ctx: Context, dev: list[Episode], sources: list[Source]) -> dict:
        """Dev documents asked with the training phrasings; evaluation phrasings stay unseen."""
        layer = self._layer(ctx)
        lp = correct = routed = count = 0
        for e, src in zip(dev, sources):
            a, b, keys = self.compiler.factors(src, ctx.device)
            ctx.hooks.set(self._state(ctx, a, b))
            for q in e.train_questions:
                logits = ctx.model(input_ids=ctx.prompt.ids(ctx.tok, q.text, device=ctx.device),
                                   use_cache=False).logits[:, -1].float()
                target = answer_ids(ctx.tok, q.gold, ctx.chat)[0]
                lp += float(F.log_softmax(logits, dim=-1)[0, target])
                correct += int(int(logits.argmax(-1)) == target)
                if "slot" in q.meta:
                    route = F.normalize(ctx.hooks.last_input[layer], dim=-1) @ F.normalize(keys, dim=-1).T
                    routed += int(int(route.argmax(-1)) == q.meta["slot"])
                count += 1
        ctx.hooks.clear()
        return {"mean_gold_logp": lp / count, "first_token_accuracy": correct / count,
                "route_accuracy": routed / count}

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": dataclasses.asdict(self.p),
                    "state_dict": {k: v.cpu() for k, v in self.compiler.state_dict().items()}}, path)

    def load(self, path: str | Path, ctx: Context) -> None:
        blob = torch.load(path, map_location="cpu")
        self.p = self.Params(**blob["params"])
        d_in, d_out = ctx.adapter.dims(self._layer(ctx))
        self.compiler = Compiler(d_in, d_out, self.p.key_rank, self.p.value_rank, self.p.initial_gain)
        self.compiler.load_state_dict(blob["state_dict"])
        self.compiler.to(ctx.device).eval()
