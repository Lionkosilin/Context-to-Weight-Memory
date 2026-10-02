#!/usr/bin/env python3
"""提问：对同一批题跑三条臂，每题导出 [INPUT]/[OUTPUT]/[REFERENCE] 的 .txt，由 agent 打分。不判分。

  none     不加载 ΔW、不给原文（闭卷）
  write    加载 outputs/<run_id>/delta.pt，不给原文（主臂）
  context  不加载 ΔW，原文放进提示（上限）
题库是 JSON：{"template": "...{question}...", "questions": [{"id", "split", "question", "gold", "aliases", "evidence"?}]}。
split 取 check（调参）、blind（超参冻结后只跑一次）或 purity（与文档无关的题，只跑 none/write 两臂，检查骨干是否受损）。
带 evidence 的题在加载时断言：evidence 逐字出现在文档中，gold 逐字出现在 evidence 中。
产物：outputs/<run_id>/<split>/<arm>/<question_id>.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default="check", choices=["check", "blind", "purity", "all"])
    p.add_argument("--arms", nargs="+", default=["none", "write", "context"])
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-4B-Base")
    p.add_argument("--doc", type=Path, required=True, help="写入时用的同一篇文档；context 臂把它放进提示")
    p.add_argument("--qa", type=Path, required=True, help="题库 JSON")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_questions(path: Path, split: str, doc: str) -> tuple[str, list[dict]]:
    """返回题库自带的提问模板和题目；带 evidence 的题断言证据在文档中。"""
    bank = json.loads(path.read_text(encoding="utf-8"))
    qs = [q for q in bank["questions"] if split == "all" or q["split"] == split]
    for q in qs:
        if "evidence" in q:
            assert q["evidence"] in doc and q["gold"] in q["evidence"], q["id"]
    return bank["template"], qs


def set_delta(model, delta: dict | None, sign: float) -> None:
    if not delta:
        return
    with torch.no_grad():
        for name, d in delta.items():
            i = int(name.split(".")[1])
            model.model.layers[i].mlp.down_proj.weight.add_(d.to(model.device), alpha=sign)


@torch.inference_mode()
def answer(model, tok, prompt: str, max_new_tokens: int) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def dump(path: Path, prompt: str, output: str, q: dict) -> None:
    ref = q["gold"] + ("；别名：" + "、".join(q["aliases"]) if q["aliases"] else "")
    if "evidence" in q:
        ref += f"\n证据：{q['evidence']}"
    path.write_text(f"[INPUT]\n{prompt}\n\n[OUTPUT]\n{output}\n\n[REFERENCE]\n{ref}\n", encoding="utf-8")


def main() -> None:
    a = parse_args()
    run_dir = a.out / a.run_id
    delta = torch.load(run_dir / "delta.pt") if "write" in a.arms else None
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    doc = a.doc.read_text(encoding="utf-8")
    template, questions = load_questions(a.qa, a.split, doc)
    if a.limit:
        questions = questions[: a.limit]
    for arm in a.arms:
        arm_dir = run_dir / a.split / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        set_delta(model, delta, +1.0 if arm == "write" else 0.0)
        for q in questions:
            prompt = template.format(question=q["question"])
            if arm == "context":
                prompt = "参考文档：\n" + doc + "\n\n" + prompt
            out = answer(model, tok, prompt, a.max_new_tokens)
            dump(arm_dir / f"{q['id']}.txt", prompt, out, q)
            first = out.strip().splitlines()[0] if out.strip() else ""
            print(f"[{arm}] {q['id']}: {first!r}", flush=True)
        set_delta(model, delta, -1.0 if arm == "write" else 0.0)
    print("out:", run_dir / a.split)


if __name__ == "__main__":
    main()
