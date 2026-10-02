"""Run configuration: YAML file + command-line overrides (--set writer.params.lr=1e-3)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "run",
    "seed": 7,
    "imports": [],                 # extra .py files or modules that register writers/tasks
    "output": "outputs/{name}/seed{seed}.json",
    "model": {
        "id": "Qwen/Qwen3-0.6B",   # Hugging Face id or local path
        "device": "auto",          # auto | cuda | cuda:1 | mps | cpu
        "dtype": "auto",           # auto | bfloat16 | float16 | float32
        "trust_remote_code": False,
        "layers_path": None,       # e.g. model.layers; null = detect
        "out_proj": None,          # e.g. mlp.down_proj; null = detect
    },
    "prompt": {
        "system": "Answer with only the exact requested value, without explanation.",
        "chat_template": "auto",
        "context_label": "Reference record:",
        "question_label": "Question:",
        "answer_label": "Answer:",
        "enable_thinking": False,
    },
    "memory": {
        "layers": [0.89],          # see ctw.adapter.resolve_layers
        "state_dtype": "bfloat16", # dtype ΔW is stored in before reading
    },
    "writer": {"name": "acwc", "params": {}, "save": None, "load": None},
    "task": {"name": "synthetic_kv", "params": {}},
    "stats": {                     # key second moment Σ = E[z zᵀ] per memory layer (ctw.stats)
        "source": "sample",        # sample: text the model generates | text: the files below
        "text": [],
        "tokens": 65536,
        "chunk": 512,
        "batch": 8,                # sequences sampled at once
        "shrink": 0.1,             # 1.0 replaces Σ with an isotropic matrix of the same trace
        "cache": "outputs/stats/{model}",
    },
    "eval": {
        "arms": ["off", "context", "write", "wrong", "random", "random_keys", "random_values"],
        "scales": [1.0],
        "max_new_tokens": 8,
        "export_txt": False,
        "heldout_text": None,      # path; reports perplexity under off and write
        "selectivity": False,      # log10 ‖ΔW z_q‖² / E‖ΔW z‖² per question; needs key statistics
    },
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_set(cfg: dict, assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    node = cfg
    *path, last = key.split(".")
    for part in path:
        node = node.setdefault(part, {})
    node[last] = yaml.safe_load(raw)


def load_config(path: str | Path | None, sets: list[str] | None = None) -> dict:
    cfg = DEFAULTS
    if path:
        cfg = merge(DEFAULTS, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        apply_set(cfg, s)
    return cfg


def import_plugins(cfg: dict) -> None:
    """Import the files and modules listed under `imports` so their @register calls run."""
    import importlib
    import importlib.util

    for entry in cfg.get("imports") or []:
        if str(entry).endswith(".py"):
            path = Path(entry).resolve()
            spec = importlib.util.spec_from_file_location(f"ctw_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            importlib.import_module(entry)
