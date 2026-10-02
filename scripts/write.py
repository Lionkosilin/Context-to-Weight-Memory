#!/usr/bin/env python3
"""写入：读一篇文档，把它写进冻结骨干上的 ΔW。

产物：outputs/<run_id>/delta.pt（BF16 ΔW）和 write.json（超参、层号、‖ΔW‖、heldout 文本上写入前后的 PPL）。
写入方法在 src/rlm/methods.py 注册，只接收原始文本。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rlm.methods import METHODS  # noqa: E402
from rlm.writer import DownWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--method", default="dcd", choices=sorted(METHODS))
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-4B-Base")
    p.add_argument("--doc", type=Path, required=True, help="要写入的文档（UTF-8 纯文本）")
    p.add_argument("--heldout", type=Path, required=True, help="不写入的文本，只用来测写入前后的 PPL")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--pick", default="last", choices=["first", "last", "mid"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--passes", type=int, default=8)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--recent", type=int, default=2048, help="教师额外看到的前文 token 数")
    p.add_argument("--prefix", type=int, default=512, help="ttcd：教师固定前缀")
    p.add_argument("--hidden-weight", type=float, default=1.0, help="dcd：隐藏状态 L1 项权重")
    return p.parse_args()


@torch.inference_mode()
def perplexity(model, tok, text: str) -> float:
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    return math.exp(float(model(input_ids=ids, labels=ids).loss))


def main() -> None:
    a = parse_args()
    run_dir = a.out / a.run_id
    assert not (run_dir / "delta.pt").exists(), f"{run_dir} 已有 ΔW，换 run_id"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    heldout = a.heldout.read_text(encoding="utf-8")
    writer = DownWriter.attach(model, n_layers=a.n_layers, lr=a.lr, pick=a.pick)
    hp = vars(a)
    ppl_before = perplexity(model, tok, heldout)
    t0 = time.time()
    info = METHODS[a.method](writer, tok, a.doc.read_text(encoding="utf-8"), hp)
    ppl_after = perplexity(model, tok, heldout)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16).cpu() for k, v in writer.delta().items()}, run_dir / "delta.pt")
    report = {"args": {k: str(v) for k, v in hp.items()}, "layers": writer.layer_ids, **info,
              "seconds": round(time.time() - t0), "delta_norm": writer.delta_norm(),
              "delta_rel_norm": writer.weight_rel_norm(), "ppl_before": ppl_before, "ppl_after": ppl_after,
              "ppl_rise": ppl_after / ppl_before - 1}
    (run_dir / "write.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("args", "log")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
