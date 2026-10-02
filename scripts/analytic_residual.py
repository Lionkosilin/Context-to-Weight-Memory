#!/usr/bin/env python3
"""Analytic residual writing: compile raw context into low-rank MLP deltas in closed form.

The writer contract is deliberately narrow:

    raw document + fixed document-independent probes -> BF16 LoRA factors

It receives no evaluation question, answer, document-specific cloze, extracted
fact table, or retrieval state.  At read time the document and probe KV states
are absent; only the question and the persistent low-rank factors remain.

For a probe suffix P, a frozen teacher reads [document; P] and a frozen student
reads P alone.  At selected MLP down projections we capture the student's gated
activation X and the teacher/student output residual R.  The per-layer write is
the dual ridge solution

    dW = R (X'X + lambda I)^-1 X' = A B,

which is already low rank because the fixed probe-token budget is small.  This
tests whether pretrained activation geometry is sufficient to transfer a raw
context residual to question-only reading without meta-training.  It is a
mechanistic baseline, not a claim that local hidden-state matching must solve
closed-book QA.

The built-in synthetic fixture has a matched decoy document.  Controls include adapter-off, full-context oracle, wrong-document
adapter, and same-rank/same-Frobenius-norm random factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3-4B"
DEFAULT_OUT = ROOT / "outputs" / "analytic_residual" / "analytic_residual.json"

# These probes are fixed before either document is constructed.  They contain
# no fixture entity, value, question, or answer.
UNIVERSAL_PROBES = (
    "Read the preceding passage carefully and retain its exact details for later use.",
    "Represent the concrete names, phrases, codes, quantities, locations, and dates in the preceding passage.",
    "Preserve every factual relation in the preceding passage so that differently worded questions can be answered later.",
    "Form a compact internal record of the preceding passage without adding unsupported information.",
)

SYSTEM = (
    "Answer with only the exact requested value. "
    "If the information is unavailable, answer UNKNOWN."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--rank", type=int, default=16,
                        help="maximum residual samples/factor rank per layer")
    parser.add_argument("--ridge", type=float, default=1e-2,
                        help="lambda relative to mean diagonal of X'X")
    parser.add_argument("--etas", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixture_documents() -> tuple[str, str]:
    main = (
        "Aster Vale Observatory issued a sealed calibration memorandum. "
        "The navigator badge was amber. "
        "The emergency signal phrase was silent meadow. "
        "The archive crate code was Q7-LUMEN. "
        "The west chamber pressure target was 37 kilopascals. "
        "The mineral sample was stored in bay 4. "
        "The morning calibration began on Tuesday."
    )
    decoy = (
        "Aster Vale Observatory issued a sealed calibration memorandum. "
        "The navigator badge was violet. "
        "The emergency signal phrase was quiet harbor. "
        "The archive crate code was M3-CEDAR. "
        "The west chamber pressure target was 52 kilopascals. "
        "The mineral sample was stored in bay 8. "
        "The morning calibration began on Friday."
    )
    return main, decoy


def fixture_questions() -> list[dict[str, str]]:
    # Called only after both document adapters have been compiled in main().
    return [
        {
            "question": "What color was the navigator badge in the Aster Vale memorandum?",
            "gold": "amber",
            "decoy": "violet",
        },
        {
            "question": "What was the emergency signal phrase in the Aster Vale memorandum?",
            "gold": "silent meadow",
            "decoy": "quiet harbor",
        },
        {
            "question": "What was the archive crate code in the Aster Vale memorandum?",
            "gold": "Q7-LUMEN",
            "decoy": "M3-CEDAR",
        },
        {
            "question": "What was the west chamber pressure target in the Aster Vale memorandum?",
            "gold": "37 kilopascals",
            "decoy": "52 kilopascals",
        },
        {
            "question": "Where was the mineral sample stored in the Aster Vale memorandum?",
            "gold": "bay 4",
            "decoy": "bay 8",
        },
        {
            "question": "On which day did the morning calibration begin in the Aster Vale memorandum?",
            "gold": "Tuesday",
            "decoy": "Friday",
        },
    ]


def token_tensor(value) -> torch.Tensor:
    if hasattr(value, "keys"):
        return value["input_ids"]
    return value


def chat_prompt_ids(tok, question: str, device: torch.device,
                    context: str | None = None) -> torch.Tensor:
    user = ""
    if context is not None:
        user += f"Reference memorandum:\n{context}\n\n"
    user += f"Question: {question}"
    encoded = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    return token_tensor(encoded).to(device)


def encode_raw(tok, text: str, device: torch.device) -> torch.Tensor:
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


@dataclass
class Factors:
    a: torch.Tensor  # out x rank, CPU float32 while compiling
    b: torch.Tensor  # rank x in, CPU float32 while compiling
    fit_fraction: float
    delta_norm: float
    residual_norm: float
    selected_positions: list[dict]


class LowRankDownHooks:
    """Apply isolated factors without mutating backbone weights."""

    def __init__(self, model, layer_ids: list[int]):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.scale = 0.0
        self.active = False
        self.handles = []
        for layer_id in self.layer_ids:
            down = model.model.layers[layer_id].mlp.down_proj

            def hook(_module, inputs, output, idx=layer_id):
                if not self.active or self.scale == 0.0 or idx not in self.state:
                    return output
                a, b = self.state[idx]
                x = inputs[0]
                # Factors persist in BF16, matching the actual state contract.
                projected = F.linear(x, b)
                delta = F.linear(projected, a)
                return output + self.scale * delta.to(output.dtype)

            self.handles.append(down.register_forward_hook(hook))

    def set_state(self, factors: dict[int, Factors], scale: float,
                  only_layers: set[int] | None = None) -> None:
        self.state = {}
        for idx, item in factors.items():
            if only_layers is not None and idx not in only_layers:
                continue
            down = self.model.model.layers[idx].mlp.down_proj
            self.state[idx] = (
                item.a.to(device=down.weight.device, dtype=torch.bfloat16),
                item.b.to(device=down.weight.device, dtype=torch.bfloat16),
            )
        self.scale = float(scale)
        self.active = bool(self.state) and self.scale != 0.0

    def clear(self) -> None:
        self.state = {}
        self.active = False
        self.scale = 0.0

    def close(self) -> None:
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_down(model, ids: torch.Tensor, layer_ids: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    captured: dict[int, list[torch.Tensor]] = {}
    handles = []
    for layer_id in layer_ids:
        down = model.model.layers[layer_id].mlp.down_proj

        def pre_hook(_module, inputs, idx=layer_id):
            captured.setdefault(idx, [None, None])[0] = inputs[0].detach().float().cpu()

        def out_hook(_module, _inputs, output, idx=layer_id):
            captured.setdefault(idx, [None, None])[1] = output.detach().float().cpu()

        handles.append(down.register_forward_pre_hook(pre_hook))
        handles.append(down.register_forward_hook(out_hook))
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_ids):
        raise RuntimeError("failed to capture every requested MLP layer")
    return {idx: (pair[0], pair[1]) for idx, pair in captured.items()}


def factor_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    ata = a.T @ a
    bbt = b @ b.T
    value = torch.sum(ata * bbt).clamp_min(0.0)
    return float(torch.sqrt(value))


def ridge_factors(x: torch.Tensor, residual: torch.Tensor, ridge: float,
                  positions: list[dict]) -> Factors:
    """Solve dW X = residual in dual form without materializing dW."""
    # x: in x rank; residual: out x rank
    gram = x.T @ x
    scale = max(float(torch.diagonal(gram).mean()), 1e-8)
    regularized = gram + (ridge * scale) * torch.eye(gram.shape[0])
    inv = torch.linalg.solve(regularized, torch.eye(gram.shape[0]))
    a = residual @ inv
    b = x.T.contiguous()
    prediction = a @ (b @ x)
    before = float(torch.sum(residual * residual))
    after = float(torch.sum((residual - prediction) ** 2))
    fit = 0.0 if before <= 0 else 1.0 - after / before
    return Factors(
        a=a.contiguous(),
        b=b,
        fit_fraction=fit,
        delta_norm=factor_norm(a, b),
        residual_norm=math.sqrt(before),
        selected_positions=positions,
    )


def compile_context(model, tok, document: str, probes: tuple[str, ...],
                    layer_ids: list[int], rank: int, ridge: float,
                    device: torch.device) -> dict[int, Factors]:
    """Writer API: its arguments deliberately contain no evaluation artifact."""
    per_layer: dict[int, list[tuple[torch.Tensor, torch.Tensor, dict]]] = {
        idx: [] for idx in layer_ids
    }
    doc_ids = encode_raw(tok, document, device)
    for probe_index, probe in enumerate(probes):
        suffix_ids = encode_raw(tok, "\n\n" + probe, device)
        teacher_ids = torch.cat([doc_ids, suffix_ids], dim=1)
        teacher = capture_down(model, teacher_ids, layer_ids)
        student = capture_down(model, suffix_ids, layer_ids)
        n = suffix_ids.shape[1]
        for idx in layer_ids:
            student_x, student_y = student[idx]
            _teacher_x, teacher_y = teacher[idx]
            sx = student_x[0, -n:]
            residual = teacher_y[0, -n:] - student_y[0, -n:]
            token_ids = suffix_ids[0].cpu().tolist()
            for token_position in range(n):
                meta = {
                    "probe": probe_index,
                    "token_position": token_position,
                    "token_id": int(token_ids[token_position]),
                    "residual_norm": float(residual[token_position].norm()),
                }
                per_layer[idx].append((sx[token_position], residual[token_position], meta))

    result: dict[int, Factors] = {}
    for idx in layer_ids:
        candidates = sorted(
            per_layer[idx], key=lambda item: item[2]["residual_norm"], reverse=True
        )[:rank]
        if not candidates:
            raise RuntimeError(f"no residual samples for layer {idx}")
        x = torch.stack([item[0] for item in candidates], dim=1)
        residual = torch.stack([item[1] for item in candidates], dim=1)
        result[idx] = ridge_factors(
            x, residual, ridge, [item[2] for item in candidates]
        )
    return result


def random_control(factors: dict[int, Factors], seed: int) -> dict[int, Factors]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for idx, item in factors.items():
        a = torch.randn(item.a.shape, generator=generator)
        b = torch.randn(item.b.shape, generator=generator)
        norm = factor_norm(a, b)
        if norm > 0:
            a.mul_(item.delta_norm / norm)
        out[idx] = Factors(
            a=a,
            b=b,
            fit_fraction=0.0,
            delta_norm=factor_norm(a, b),
            residual_norm=item.residual_norm,
            selected_positions=[],
        )
    return out


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[\s\"'`]+|[\s\"'`.,;:!?]+$", "", text)
    return " ".join(text.split())


@torch.inference_mode()
def answer_logprob(model, tok, prompt_ids: torch.Tensor, answer: str) -> dict[str, float]:
    answer_ids = encode_raw(tok, answer, prompt_ids.device)
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(input_ids=full, use_cache=False).logits.float()
    start = prompt_ids.shape[1] - 1
    selected = logits[:, start:start + answer_ids.shape[1]]
    logp = F.log_softmax(selected, dim=-1)
    token_logp = logp.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    return {
        "sum_logp": float(token_logp.sum()),
        "mean_logp": float(token_logp.mean()),
    }


@torch.inference_mode()
def evaluate(model, tok, questions: list[dict[str, str]], device: torch.device,
             max_new_tokens: int, context: str | None = None) -> dict:
    rows = []
    for item in questions:
        prompt_ids = chat_prompt_ids(tok, item["question"], device, context=context)
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        reply = tok.decode(
            generated[0, prompt_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        gold_lp = answer_logprob(model, tok, prompt_ids, item["gold"])
        decoy_lp = answer_logprob(model, tok, prompt_ids, item["decoy"])
        normalized = normalize_answer(reply)
        rows.append({
            **item,
            "reply": reply,
            "exact": normalized == normalize_answer(item["gold"]),
            "contains": normalize_answer(item["gold"]) in normalized,
            "decoy_contains": normalize_answer(item["decoy"]) in normalized,
            "gold_sum_logp": gold_lp["sum_logp"],
            "gold_mean_logp": gold_lp["mean_logp"],
            "decoy_sum_logp": decoy_lp["sum_logp"],
            "gold_minus_decoy_mean_logp": (
                gold_lp["mean_logp"] - decoy_lp["mean_logp"]
            ),
            "prompt_tokens": int(prompt_ids.shape[1]),
        })
    return {
        "exact": sum(int(row["exact"]) for row in rows),
        "contains": sum(int(row["contains"]) for row in rows),
        "decoy_contains": sum(int(row["decoy_contains"]) for row in rows),
        "mean_gold_logp": sum(row["gold_mean_logp"] for row in rows) / len(rows),
        "mean_gold_minus_decoy_logp": (
            sum(row["gold_minus_decoy_mean_logp"] for row in rows) / len(rows)
        ),
        "rows": rows,
    }


def factor_summary(factors: dict[int, Factors]) -> dict:
    return {
        str(idx): {
            "rank": int(item.a.shape[1]),
            "a_shape": list(item.a.shape),
            "b_shape": list(item.b.shape),
            "fit_fraction": item.fit_fraction,
            "delta_frobenius_norm": item.delta_norm,
            "teacher_student_residual_norm": item.residual_norm,
            "selected_positions": item.selected_positions,
        }
        for idx, item in factors.items()
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.device
    n_layers = len(model.model.layers)
    if any(idx < 0 or idx >= n_layers for idx in args.layers):
        raise ValueError(f"layers must lie in 0..{n_layers - 1}")
    if args.rank <= 0:
        raise ValueError("rank must be positive")

    main_document, decoy_document = fixture_documents()
    t0 = time.time()
    # Crucial ordering: both writes finish before evaluation items are created.
    main_factors = compile_context(
        model, tok, main_document, UNIVERSAL_PROBES,
        args.layers, args.rank, args.ridge, device,
    )
    decoy_factors = compile_context(
        model, tok, decoy_document, UNIVERSAL_PROBES,
        args.layers, args.rank, args.ridge, device,
    )
    random_factors = random_control(main_factors, args.seed + 991)
    write_seconds = time.time() - t0
    questions = fixture_questions()

    # Runtime leakage assertions are intentionally redundant and visible.
    for item in questions:
        if normalize_answer(item["gold"]) in normalize_answer(item["question"]):
            raise RuntimeError("an evaluation question contains its gold answer")
        if normalize_answer(item["decoy"]) in normalize_answer(item["question"]):
            raise RuntimeError("an evaluation question contains its decoy answer")

    hooks = LowRankDownHooks(model, args.layers)
    report = {
        "method": "universal_probe_analytic_context_residual_lora",
        "status": "development_only",
        "writer_contract": {
            "inputs": ["raw_document", "fixed_document_independent_probe_bank"],
            "forbidden_inputs": [
                "evaluation_questions", "evaluation_answers", "document_specific_clozes",
                "fact_table", "retrieval_index", "reader_kv_cache",
            ],
            "persistent_state": "BF16 low-rank factors on selected MLP down projections",
            "reader_input": "question only",
        },
        "model_dir": str(args.model_dir),
        "model_type": str(model.config.model_type),
        "layers": args.layers,
        "rank": args.rank,
        "ridge": args.ridge,
        "etas": args.etas,
        "seed": args.seed,
        "probe_bank_sha256": sha256_text("\n".join(UNIVERSAL_PROBES)),
        "main_document_sha256": sha256_text(main_document),
        "decoy_document_sha256": sha256_text(decoy_document),
        "write_seconds_for_two_documents": write_seconds,
        "main_factors": factor_summary(main_factors),
        "decoy_factors": factor_summary(decoy_factors),
        "random_factors": factor_summary(random_factors),
        "evaluations": {},
    }
    try:
        hooks.clear()
        report["evaluations"]["off"] = evaluate(
            model, tok, questions, device, args.max_new_tokens
        )
        report["evaluations"]["full_context_oracle"] = evaluate(
            model, tok, questions, device, args.max_new_tokens,
            context=main_document,
        )

        states = {
            "main_all": (main_factors, None),
            "wrong_document_all": (decoy_factors, None),
            "random_same_rank_norm_all": (random_factors, None),
        }
        states.update({
            f"main_layer_{idx}": (main_factors, {idx}) for idx in args.layers
        })
        for eta in args.etas:
            for name, (factors, only_layers) in states.items():
                hooks.set_state(factors, eta, only_layers=only_layers)
                key = f"{name}_eta_{eta:g}"
                result = evaluate(
                    model, tok, questions, device, args.max_new_tokens
                )
                report["evaluations"][key] = result
                print(
                    f"{key:42s} exact={result['exact']}/{len(questions)} "
                    f"contains={result['contains']}/{len(questions)} "
                    f"gold_logp={result['mean_gold_logp']:+.3f} "
                    f"gold-decoy={result['mean_gold_minus_decoy_logp']:+.3f}"
                )
    finally:
        hooks.close()

    off = report["evaluations"]["off"]
    for result in report["evaluations"].values():
        result["delta_mean_gold_logp_vs_off"] = (
            result["mean_gold_logp"] - off["mean_gold_logp"]
        )
    ranked = sorted(
        ((name, result["contains"], result["delta_mean_gold_logp_vs_off"])
         for name, result in report["evaluations"].items()
         if name not in {"off", "full_context_oracle"}),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    report["exploratory_ranking"] = [
        {"name": name, "contains": contains, "delta_mean_gold_logp_vs_off": delta}
        for name, contains, delta in ranked
    ]
    report["elapsed_seconds"] = time.time() - t0

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== gates ===")
    print(f"off                 {off['contains']}/{len(questions)}")
    oracle = report["evaluations"]["full_context_oracle"]
    print(f"full-context oracle {oracle['contains']}/{len(questions)}")
    if ranked:
        print(f"best adapter         {ranked[0][0]}: {ranked[0][1]}/{len(questions)} "
              f"dlogp={ranked[0][2]:+.3f}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
