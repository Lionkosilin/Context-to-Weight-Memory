"""The full pipeline: load → (fit) → write every evaluation document → question it under each arm."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from . import budget
from .adapter import ModelAdapter
from .device import DTYPES, load_model
from .eval import evaluate_split, write_split
from .memory import MemoryHooks
from .prompting import PromptFormat, encode
from .stats import StatsCache
from .tasks import build_task
from .writers import Context, build_writer


def setup(cfg: dict):
    """Model, tokenizer, adapter, hooks, and writer context from a config."""
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    m = cfg["model"]
    model, tok = load_model(m["id"], m["device"], m["dtype"], m["trust_remote_code"])
    return context_for(model, tok, cfg)


def context_for(model, tok, cfg: dict) -> Context:
    adapter = ModelAdapter.from_model(model, cfg["model"]["layers_path"], cfg["model"]["out_proj"])
    layers = adapter.resolve_layers(cfg["memory"]["layers"])
    hooks = MemoryHooks(adapter, layers)
    return Context(model=model, tok=tok, adapter=adapter, hooks=hooks, layers=layers,
                   prompt=PromptFormat(**cfg["prompt"]), seed=cfg["seed"],
                   stats=StatsCache(model, tok, adapter, hooks, cfg["stats"], cfg["model"]["id"]))


def output_path(cfg: dict) -> Path:
    return Path(cfg["output"].format(name=cfg["name"], seed=cfg["seed"]))


def run(cfg: dict, ctx: Context | None = None) -> dict:
    t0 = time.time()
    ctx = ctx or setup(cfg)
    task = build_task(cfg["task"]["name"], cfg["task"]["params"], ctx.tok, cfg["seed"])
    writer = build_writer(cfg["writer"]["name"], cfg["writer"]["params"])
    fit_report = {}
    if cfg["writer"].get("load"):
        writer.load(cfg["writer"]["load"], ctx)
    elif writer.trainable:
        if not task.fit_splits:
            raise ValueError(f"writer {writer.name!r} needs a task with training splits")
        train, dev = task.fit_splits
        print(f"fitting {writer.name} on {len(task.splits[train])} documents", flush=True)
        fit_report = writer.fit(ctx, task.splits[train], task.splits[dev])
    if cfg["writer"].get("save"):
        writer.save(cfg["writer"]["save"].format(name=cfg["name"], seed=cfg["seed"]))
    fit_seconds = time.time() - t0

    e = cfg["eval"]
    heldout = None
    if e.get("heldout_text"):
        heldout = encode(ctx.tok, Path(e["heldout_text"]).read_text(encoding="utf-8"), ctx.device)
    export_root = output_path(cfg).with_suffix("") if e.get("export_txt") else None
    state_dtype = DTYPES[cfg["memory"]["state_dtype"]]
    evaluations, state_params = {}, 0
    for k, split in enumerate(task.eval_splits):
        episodes = task.splits[split]
        states = write_split(ctx, writer, episodes, e["arms"], cfg["seed"] + 50000 + 10000 * k, state_dtype)
        state_params = states["write"][0].nbytes(1)
        evaluations[split] = evaluate_split(
            ctx, episodes, states, e["arms"], e["scales"], e["max_new_tokens"],
            export_dir=export_root / split if export_root else None, heldout=heldout,
            selectivity=e["selectivity"])
        line = "  ".join(f"{a}={r['contains']}/{r['n']}" for a, r in evaluations[split].items())
        print(f"{split}: {line}", flush=True)

    report = {
        "config": cfg,
        "model": ctx.adapter.describe(),
        "layers": ctx.layers,
        "state_params_per_document": state_params,
        "state_bytes_per_document": state_params * (torch.finfo(state_dtype).bits // 8),
        "kv_equivalent_tokens": budget.kv_equivalent_tokens(ctx.adapter, state_params),
        "fit": fit_report,
        "fit_seconds": fit_seconds,
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - t0,
    }
    out = output_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return report
