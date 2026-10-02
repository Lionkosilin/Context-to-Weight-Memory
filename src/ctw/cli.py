"""ctw command line.

  ctw inspect --model Qwen/Qwen3-0.6B            what the adapter sees, and state sizes
  ctw run configs/qwen3-0.6b-acwc.yaml           fit, write, and evaluate
  ctw write CONFIG --document doc.txt --out s.safetensors
  ctw ask CONFIG --state s.safetensors --question "..."
  ctw summarize outputs/x/seed*.json --out agg.json
  ctw list                                       registered writers and tasks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _config(args):
    from .config import import_plugins, load_config
    cfg = load_config(args.config, args.set)
    import_plugins(cfg)
    return cfg


def cmd_inspect(args) -> None:
    from . import budget
    from .adapter import ModelAdapter
    from .device import load_model

    model, _ = load_model(args.model, args.device, args.dtype, args.trust_remote_code)
    adapter = ModelAdapter.from_model(model, args.layers_path, args.out_proj)
    layers = adapter.resolve_layers(args.layers)
    info = adapter.describe()
    info["device"] = str(adapter.device)
    info["dtype"] = str(next(model.parameters()).dtype)
    info["selected_layers"] = layers
    dense = budget.dense_state_params(adapter, layers)
    low = budget.lowrank_state_params(adapter, layers, args.rank)
    info["kv_params_per_token"] = budget.kv_params_per_token(model.config)
    info["dense_state_params"] = dense
    info["dense_state_kv_tokens"] = budget.kv_equivalent_tokens(adapter, dense)
    info[f"rank{args.rank}_state_params"] = low
    info[f"rank{args.rank}_state_kv_tokens"] = budget.kv_equivalent_tokens(adapter, low)
    print(json.dumps(info, indent=2))


def cmd_run(args) -> None:
    from .run import run
    run(_config(args))


def _writer_ready(cfg, ctx):
    from .writers import build_writer

    writer = build_writer(cfg["writer"]["name"], cfg["writer"]["params"])
    if writer.trainable:
        if not cfg["writer"].get("load"):
            raise SystemExit(f"{writer.name} needs a fitted file: --set writer.load=PATH "
                             "(produced by `ctw run` with writer.save=PATH)")
        writer.load(cfg["writer"]["load"], ctx)
    return writer


def cmd_write(args) -> None:
    from .device import DTYPES
    from .run import setup

    cfg = _config(args)
    ctx = setup(cfg)
    writer = _writer_ready(cfg, ctx)
    state = writer.write(ctx, Path(args.document).read_text(encoding="utf-8"))
    state = state.to(dtype=DTYPES[cfg["memory"]["state_dtype"]])
    state.meta.update({"model": cfg["model"]["id"], "layers": ctx.layers})
    state.save(args.out)
    print(f"wrote {args.out}  layers={ctx.layers}  params={state.nbytes(1)}  norm={state.frobenius():.3f}")


def cmd_ask(args) -> None:
    from .eval import generate
    from .memory import MemoryState
    from .run import setup

    cfg = _config(args)
    ctx = setup(cfg)
    state = MemoryState.load(args.state)
    prompt = ctx.prompt.ids(ctx.tok, args.question, device=ctx.device)
    for label, s in (("without memory", None), ("with memory", state)):
        ctx.hooks.set(s, scale=args.scale)
        print(f"[{label}] {generate(ctx, prompt, args.max_new_tokens)}")
    ctx.hooks.clear()


def cmd_summarize(args) -> None:
    from .summary import summarize

    out = summarize([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, s in out["splits"].items():
        print(split + ":  " + "  ".join(f"{a}={v['contains']}/{v['n']}" for a, v in s["arms"].items()))
    print(f"wrote {args.out}")


def cmd_list(_args) -> None:
    import dataclasses

    from .tasks import TASKS
    from .writers import WRITERS

    for name, cls in sorted(WRITERS.items()):
        defaults = {f.name: (f.default if f.default is not dataclasses.MISSING else "...")
                    for f in dataclasses.fields(cls.Params)}
        print(f"writer {name}: {defaults}")
    for name in sorted(TASKS):
        print(f"task   {name}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ctw", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="show layers, projection, and state sizes for a model")
    i.add_argument("--model", required=True)
    i.add_argument("--device", default="auto")
    i.add_argument("--dtype", default="auto")
    i.add_argument("--trust-remote-code", action="store_true")
    i.add_argument("--layers-path")
    i.add_argument("--out-proj")
    i.add_argument("--layers", nargs="+", default=["0.89"], help="layer spec, e.g. 24  -4  0.89  last:8")
    i.add_argument("--rank", type=int, default=4)
    i.set_defaults(fn=cmd_inspect)

    def with_config(sp):
        sp.add_argument("config", nargs="?", help="YAML config; defaults apply when omitted")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set writer.params.lr=1e-3")
        return sp

    r = with_config(sub.add_parser("run", help="fit, write, and evaluate"))
    r.set_defaults(fn=cmd_run)

    w = with_config(sub.add_parser("write", help="write one document into a state file"))
    w.add_argument("--document", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_write)

    a = with_config(sub.add_parser("ask", help="ask a question with and without a state file"))
    a.add_argument("--state", required=True)
    a.add_argument("--question", required=True)
    a.add_argument("--scale", type=float, default=1.0)
    a.add_argument("--max-new-tokens", type=int, default=32)
    a.set_defaults(fn=cmd_ask)

    s = sub.add_parser("summarize", help="aggregate runs that differ only in seed")
    s.add_argument("reports", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_summarize)

    sub.add_parser("list", help="registered writers and tasks").set_defaults(fn=cmd_list)

    args = p.parse_args(argv)
    if hasattr(args, "layers") and isinstance(args.layers, list):
        args.layers = [_num(x) for x in args.layers]
    args.fn(args)


def _num(x: str):
    try:
        return float(x) if "." in x else int(x)
    except ValueError:
        return x


if __name__ == "__main__":
    main()
