"""Write every document of a split, then question it under the control arms.

Arms
  off      no ΔW, no document              what the backbone already knows
  context  no ΔW, document in the prompt   whether the question is answerable
  write    the document's ΔW               main arm
  wrong    the decoy document's ΔW         does the answer follow the weight's document
  random   random ΔW, same shape and norm  does any perturbation get lucky
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .memory import MemoryState
from .prompting import answer_ids
from .tasks import Episode, Question
from .writers import Context, Writer

ARMS = ("off", "context", "write", "wrong", "random")


def normalize(text: str) -> str:
    text = re.sub(r"^[\s\"'`]+|[\s\"'`.,;:!?]+$", "", text.strip().lower())
    return " ".join(text.split())


@torch.no_grad()
def generate(ctx: Context, prompt: torch.Tensor, max_new_tokens: int) -> str:
    pad = ctx.tok.pad_token_id if ctx.tok.pad_token_id is not None else ctx.tok.eos_token_id
    out = ctx.model.generate(prompt, attention_mask=torch.ones_like(prompt),
                             max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
    return ctx.tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def gold_logprob(ctx: Context, prompt: torch.Tensor, gold: str) -> float:
    """Mean log-probability of the gold answer tokens after the prompt."""
    ans = torch.tensor([answer_ids(ctx.tok, gold, ctx.chat)], device=prompt.device)
    logits = ctx.model(input_ids=torch.cat([prompt, ans], dim=1), use_cache=False).logits.float()
    start = prompt.shape[1] - 1
    logp = F.log_softmax(logits[:, start:start + ans.shape[1]], dim=-1)
    return float(logp.gather(-1, ans.unsqueeze(-1)).mean())


@torch.no_grad()
def perplexity(ctx: Context, ids: torch.Tensor) -> float:
    return math.exp(float(ctx.model(input_ids=ids, labels=ids, use_cache=False).loss))


def _score(reply: str, q: Question) -> dict:
    norm = normalize(reply)
    golds = [normalize(g) for g in [q.gold, *q.aliases]]
    return {
        "exact": norm in golds,
        "contains": any(g in norm for g in golds),
        "wrong_contains": q.wrong_gold is not None and normalize(q.wrong_gold) in norm,
    }


def _export(path: Path, prompt_text: str, reply: str, q: Question) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = q.gold + (f" | aliases: {', '.join(q.aliases)}" if q.aliases else "")
    if q.wrong_gold:
        ref += f" | decoy answer: {q.wrong_gold}"
    path.write_text(f"[INPUT]\n{prompt_text}\n\n[OUTPUT]\n{reply}\n\n[REFERENCE]\n{ref}\n", encoding="utf-8")


def write_split(ctx: Context, writer: Writer, episodes: list[Episode], arms, seed: int,
                state_dtype: torch.dtype) -> dict[str, list[MemoryState | None]]:
    """All writes for a split happen here, before any question is built."""
    def persist(doc):
        return writer.write(ctx, doc).to(dtype=state_dtype)

    states = {"write": [persist(e.document) for e in episodes]}
    if "wrong" in arms:
        states["wrong"] = [persist(e.decoy) if e.decoy is not None else None for e in episodes]
    if "random" in arms:
        states["random"] = [s.random_like(seed + k) for k, s in enumerate(states["write"])]
    return states


def evaluate_split(ctx: Context, episodes: list[Episode], states: dict, arms, scales,
                   max_new_tokens: int, export_dir: Path | None = None,
                   heldout: torch.Tensor | None = None) -> dict:
    results = {}
    plan = [(arm, 1.0) for arm in arms if arm != "write"]
    plan += [("write", s) for s in scales] if "write" in arms else []
    for arm, scale in plan:
        key = arm if arm != "write" or len(scales) == 1 else f"write@{scale:g}"
        rows, ppl = [], []
        for k, e in enumerate(episodes):
            state = states[arm][k] if arm in ("write", "wrong", "random") else None
            if arm in ("wrong", "random", "write") and state is None:
                continue
            ctx.hooks.set(state, scale=scale)
            if heldout is not None and arm in ("off", "write"):
                ppl.append(perplexity(ctx, heldout))
            for q in e.questions:
                context = e.document if arm == "context" else None
                prompt = ctx.prompt.ids(ctx.tok, q.text, context=context, device=ctx.device)
                reply = generate(ctx, prompt, max_new_tokens)
                rows.append({"episode": e.id, "question": q.id, "gold": q.gold, "reply": reply,
                             "gold_logp": gold_logprob(ctx, prompt, q.gold), **_score(reply, q)})
                if export_dir is not None:
                    _export(export_dir / key / f"{q.id}.txt",
                            ctx.tok.decode(prompt[0], skip_special_tokens=False), reply, q)
            ctx.hooks.clear()
        if not rows:
            continue
        n = len(rows)
        results[key] = {
            "n": n,
            "exact": sum(r["exact"] for r in rows),
            "contains": sum(r["contains"] for r in rows),
            "wrong_contains": sum(r["wrong_contains"] for r in rows),
            "mean_gold_logp": sum(r["gold_logp"] for r in rows) / n,
            **({"mean_heldout_ppl": sum(ppl) / len(ppl)} if ppl else {}),
            "rows": rows,
        }
    if "off" in results:
        for r in results.values():
            r["delta_gold_logp_vs_off"] = r["mean_gold_logp"] - results["off"]["mean_gold_logp"]
    return results
