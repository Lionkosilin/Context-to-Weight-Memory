#!/usr/bin/env python3
"""Associative Context Weight Compiler (ACWC): raw context -> low-rank MLP weight.

    target raw context -> persistent low-rank MLP weights -> question-only read

A shared compiler is learned on separate synthetic training documents.  For a
new target document it runs one frozen forward pass over the raw text and emits
rank-m factors A, B for one MLP down projection.  The target writer receives no
question or answer.  Outer-loop QA on training documents teaches the general
read/write interface; it supplies no target-document facts.

Each document consists of shuffled sentences of the form
"The <relation> key was <single-token value>."  Per sentence, the key source is
the mean down_proj input over the relation tokens and the value source is the
lm_head row of the final content token.  The compiler learns a diagonal plus
low-rank key metric and a gain; the value map stays fixed so that values unseen
in outer training can still be written.

Controls: adapter off, full-context oracle, wrong-document factors, and random
factors matched in rank and Frobenius norm.  Test questions are instantiated
only after every test document has been compiled and persisted.
"""

from __future__ import annotations

import argparse
import copy
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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3-4B"
DEFAULT_OUT = ROOT / "outputs" / "acwc" / "acwc.json"
SYSTEM = "Answer with only the assigned word, without explanation."

RELATIONS = (
    {
        "id": "navigation",
        "statement": "The navigation key was {value}.",
        "train_questions": (
            "What was the navigation key?",
            "Which word was assigned as the navigation key?",
            "State the navigation key.",
        ),
        "eval_question": "Report the assigned value of the navigation key.",
    },
    {
        "id": "emergency",
        "statement": "The emergency key was {value}.",
        "train_questions": (
            "What was the emergency key?",
            "Which word was assigned as the emergency key?",
            "State the emergency key.",
        ),
        "eval_question": "Report the assigned value of the emergency key.",
    },
    {
        "id": "archive",
        "statement": "The archive key was {value}.",
        "train_questions": (
            "What was the archive key?",
            "Which word was assigned as the archive key?",
            "State the archive key.",
        ),
        "eval_question": "Report the assigned value of the archive key.",
    },
    {
        "id": "calibration",
        "statement": "The calibration key was {value}.",
        "train_questions": (
            "What was the calibration key?",
            "Which word was assigned as the calibration key?",
            "State the calibration key.",
        ),
        "eval_question": "Report the assigned value of the calibration key.",
    },
)

SEEN_VALUES = (
    "amber", "silver", "golden", "blue", "green", "orange", "purple", "black",
    "white", "yellow", "red", "brown", "pink", "gray", "rabbit", "lion",
)
OOD_VALUES = ("wolf", "bear", "horse", "fox", "owl", "mouse", "cat", "dog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--train-docs", type=int, default=64)
    parser.add_argument("--dev-docs", type=int, default=8)
    parser.add_argument("--test-docs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--route-weight", type=float, default=0.2)
    parser.add_argument("--route-temperature", type=float, default=20.0)
    parser.add_argument("--key-rank", type=int, default=64,
                        help="rank of the shared relation-key metric residual")
    parser.add_argument("--value-rank", type=int, default=0,
                        help="0 keeps the tied-embedding value map fixed for OOD transfer")
    parser.add_argument("--initial-gain", type=float, default=96.0)
    parser.add_argument("--value-source", choices=("hidden", "embedding"),
                        default="embedding")
    parser.add_argument("--max-new-tokens", type=int, default=1,
                        help="controlled values are exactly one tokenizer token")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_tensor(value) -> torch.Tensor:
    return value["input_ids"] if hasattr(value, "keys") else value


def prompt_ids(tok, question: str, device: torch.device,
               context: str | None = None) -> torch.Tensor:
    user = ""
    if context is not None:
        user += f"Reference record:\n{context}\n\n"
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


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[\s\"'`]+|[\s\"'`.,;:!?]+$", "", value)
    return " ".join(value.split())


@dataclass(frozen=True)
class EpisodeSpec:
    document: str
    values: dict[str, str]
    sentence_relations: tuple[str, ...]
    seed: int


@dataclass
class ContextSource:
    key_sources: torch.Tensor       # slots x d_ff, frozen
    value_sources: torch.Tensor     # slots x d_model, frozen
    document_sha256: str
    document_tokens: int


@dataclass
class PersistedFactors:
    a: torch.Tensor                 # d_model x slots, BF16
    b: torch.Tensor                 # slots x d_ff, BF16
    delta_norm: float


def make_episode(seed: int, value_pool: tuple[str, ...]) -> EpisodeSpec:
    rng = random.Random(seed)
    sampled = rng.sample(list(value_pool), k=len(RELATIONS))
    values = {relation["id"]: value for relation, value in zip(RELATIONS, sampled)}
    order = list(range(len(RELATIONS)))
    rng.shuffle(order)
    statements = [
        RELATIONS[index]["statement"].format(value=values[RELATIONS[index]["id"]])
        for index in order
    ]
    return EpisodeSpec(
        document=" ".join(statements),
        values=values,
        sentence_relations=tuple(RELATIONS[index]["id"] for index in order),
        seed=seed,
    )


def build_specs(base_seed: int, count: int,
                value_pool: tuple[str, ...]) -> list[EpisodeSpec]:
    specs = []
    seen = set()
    cursor = base_seed
    while len(specs) < count:
        item = make_episode(cursor, value_pool)
        signature = tuple(item.values[relation["id"]] for relation in RELATIONS)
        cursor += 1
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(item)
    return specs


def encode_document_parts(tok, document: str,
                          device: torch.device) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", document.strip())
        if sentence.strip()
    ]
    if not sentences:
        raise RuntimeError("raw document contains no sentence")
    tensors = []
    offsets = []
    cursor = 0
    for sentence_index, sentence in enumerate(sentences):
        if sentence_index:
            sentence = " " + sentence
        ids = tok(
            sentence, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        tensors.append(ids)
        offsets.append((cursor, cursor + ids.shape[1]))
        cursor += ids.shape[1]
    return torch.cat(tensors, dim=1), offsets


def capture_context_source(model, tok, document: str, layer: int,
                           device: torch.device,
                           value_source: str = "embedding") -> ContextSource:
    """One raw-document forward.  No question or answer argument exists."""
    captured: dict[str, torch.Tensor] = {}
    mlp = model.model.layers[layer].mlp

    def mlp_pre(_module, inputs):
        captured["h"] = inputs[0].detach().float().cpu()

    def down_pre(_module, inputs):
        captured["z"] = inputs[0].detach().float().cpu()

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
    ]
    ids, offsets = encode_document_parts(tok, document, device)
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"h", "z"}:
        raise RuntimeError("context hooks failed")

    key_rows = []
    value_rows = []
    for start, end in offsets:
        # The final token is punctuation and the preceding token is the final
        # content token.  The relation key excludes both.  This positional
        # rule is fixed across all episodes and uses no value label.
        if end - start < 4:
            raise RuntimeError("sentence is too short for the compiler contract")
        key_rows.append(captured["z"][0, start:end - 2].mean(dim=0))
        if value_source == "hidden":
            value_rows.append(captured["h"][0, end - 2])
        elif value_source == "embedding":
            # Decode the final raw-context token and re-tokenize it without its
            # leading whitespace.  This is a document-only operation and makes
            # the source direction share the model's universal output geometry,
            # which can in principle transfer to values unseen in outer training.
            raw_piece = tok.decode([int(ids[0, end - 2])]).strip()
            standalone = tok(raw_piece, add_special_tokens=False).input_ids
            if len(standalone) != 1:
                raise RuntimeError(
                    f"final content token {raw_piece!r} is not standalone single-token"
                )
            value_rows.append(
                model.lm_head.weight[int(standalone[0])].detach().float().cpu()
            )
        else:
            raise ValueError(f"unknown value source {value_source!r}")
    return ContextSource(
        key_sources=torch.stack(key_rows),
        value_sources=torch.stack(value_rows),
        document_sha256=sha256_text(document),
        document_tokens=int(ids.shape[1]),
    )


class ContextWeightCompiler(nn.Module):
    """Shared compiler learned on support episodes, then frozen for target docs."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 key_rank: int, residual_rank: int, initial_gain: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.key_rank = key_rank
        self.residual_rank = residual_rank
        self.key_log_diagonal = nn.Parameter(torch.zeros(intermediate_size))
        self.key_left = nn.Parameter(torch.zeros(intermediate_size, key_rank))
        self.key_right = nn.Parameter(torch.empty(intermediate_size, key_rank))
        nn.init.normal_(self.key_right, std=0.005)
        self.value_left = nn.Parameter(torch.zeros(hidden_size, residual_rank))
        self.value_right = nn.Parameter(torch.empty(hidden_size, residual_rank))
        nn.init.normal_(self.value_right, std=0.02)
        self.log_gain = nn.Parameter(torch.tensor(math.log(initial_gain)))

    def factors(self, source: ContextSource,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = source.key_sources.to(device=device, dtype=torch.float32)
        values = source.value_sources.to(device=device, dtype=torch.float32)

        # A bounded diagonal metric is enough to test whether outer-loop
        # supervision can align raw statement keys with paraphrased questions.
        diagonal = torch.exp(self.key_log_diagonal.clamp(-3.0, 3.0))
        weighted_keys = keys * diagonal
        key_residual = (weighted_keys @ self.key_right) @ self.key_left.T
        transformed_keys = weighted_keys + key_residual
        b = transformed_keys / transformed_keys.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        values = F.normalize(values, dim=-1)
        residual = (values @ self.value_right) @ self.value_left.T
        transformed_values = values + residual
        gain = torch.exp(self.log_gain.clamp(math.log(1.0), math.log(1024.0)))
        a = (gain * transformed_values).T.contiguous()
        return a, b.contiguous(), transformed_keys


class DynamicLowRankHook:
    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.a: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self.last_input: torch.Tensor | None = None
        down = model.model.layers[layer].mlp.down_proj

        def hook(_module, inputs, output):
            self.last_input = inputs[0][:, -1].detach().float()
            if self.a is None or self.b is None:
                return output
            projected = F.linear(inputs[0].float(), self.b.float())
            delta = F.linear(projected, self.a.float())
            return output + delta.to(output.dtype)

        self.handle = down.register_forward_hook(hook)

    def set(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self.a, self.b = a, b

    def clear(self) -> None:
        self.a = None
        self.b = None
        self.last_input = None

    def close(self) -> None:
        self.clear()
        self.handle.remove()


def relation_slot(spec: EpisodeSpec, relation_id: str) -> int:
    return spec.sentence_relations.index(relation_id)


def answer_id(tok, value: str, device: torch.device) -> int:
    ids = tok(value, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise RuntimeError(f"controlled value {value!r} is not one token: {ids}")
    return int(ids[0])


def train_examples(tok, specs: list[EpisodeSpec], device: torch.device) -> list[dict]:
    examples = []
    for doc_index, spec in enumerate(specs):
        for relation in RELATIONS:
            for question in relation["train_questions"]:
                examples.append({
                    "doc_index": doc_index,
                    "relation": relation["id"],
                    "prompt_ids": prompt_ids(tok, question, device),
                    "answer_id": answer_id(tok, spec.values[relation["id"]], device),
                })
    return examples


def factor_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    ata = a.float().T @ a.float()
    bbt = b.float() @ b.float().T
    return float(torch.sqrt(torch.sum(ata * bbt).clamp_min(0.0)))


def persist(compiler: ContextWeightCompiler, source: ContextSource,
            device: torch.device) -> PersistedFactors:
    with torch.no_grad():
        a, b, _ = compiler.factors(source, device)
    a = a.to(dtype=torch.bfloat16).contiguous()
    b = b.to(dtype=torch.bfloat16).contiguous()
    return PersistedFactors(
        a=a,
        b=b,
        delta_norm=factor_frobenius(a, b),
    )


def random_like(state: PersistedFactors, seed: int) -> PersistedFactors:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(state.a.shape, generator=generator)
    b = torch.randn(state.b.shape, generator=generator)
    norm = factor_frobenius(a, b)
    if norm > 0:
        a.mul_(state.delta_norm / norm)
    return PersistedFactors(
        a=a.to(device=state.a.device, dtype=torch.bfloat16),
        b=b.to(device=state.b.device, dtype=torch.bfloat16),
        delta_norm=factor_frobenius(a, b),
    )


@torch.inference_mode()
def gold_logprob(model, prompt: torch.Tensor, target_id: int) -> float:
    logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float()
    return float(F.log_softmax(logits, dim=-1)[0, target_id])


@torch.inference_mode()
def generate(model, tok, prompt: torch.Tensor, max_new_tokens: int) -> str:
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(output[0, prompt.shape[1]:], skip_special_tokens=True).strip()


def make_eval_items(tok, specs: list[EpisodeSpec], device: torch.device) -> list[dict]:
    items = []
    for doc_index, spec in enumerate(specs):
        for relation in RELATIONS:
            value = spec.values[relation["id"]]
            question = relation["eval_question"]
            items.append({
                "doc_index": doc_index,
                "relation": relation["id"],
                "question": question,
                "gold": value,
                "gold_id": answer_id(tok, value, device),
                "prompt_ids": prompt_ids(tok, question, device),
                "oracle_prompt_ids": prompt_ids(tok, question, device, context=spec.document),
            })
    return items


def deranged_value_specs(specs: list[EpisodeSpec]) -> list[EpisodeSpec]:
    """Build one raw decoy document per target by rotating its distinct values."""
    relation_ids = [relation["id"] for relation in RELATIONS]
    relation_by_id = {relation["id"]: relation for relation in RELATIONS}
    out = []
    for spec in specs:
        original = [spec.values[relation_id] for relation_id in relation_ids]
        if len(set(original)) != len(original):
            raise RuntimeError("deranged control requires distinct per-document values")
        rotated = original[1:] + original[:1]
        values = dict(zip(relation_ids, rotated))
        if any(values[key] == spec.values[key] for key in relation_ids):
            raise RuntimeError("value rotation failed to derange every relation")
        document = " ".join(
            relation_by_id[relation_id]["statement"].format(
                value=values[relation_id]
            )
            for relation_id in spec.sentence_relations
        )
        out.append(EpisodeSpec(
            document=document,
            values=values,
            sentence_relations=spec.sentence_relations,
            seed=spec.seed + 900000,
        ))
    return out


def evaluate_condition(model, tok, hook: DynamicLowRankHook,
                       items: list[dict], specs: list[EpisodeSpec],
                       states: list[PersistedFactors] | None,
                       max_new_tokens: int, condition: str,
                       wrong_specs: list[EpisodeSpec] | None = None) -> dict:
    rows = []
    for item in items:
        source_doc = item["doc_index"]
        prompt = item["prompt_ids"]
        wrong_value = None
        if condition == "off":
            hook.clear()
        elif condition == "oracle":
            hook.clear()
            prompt = item["oracle_prompt_ids"]
        else:
            if states is None:
                raise RuntimeError("adapter condition has no states")
            state_index = source_doc
            if wrong_specs is not None:
                wrong_value = wrong_specs[source_doc].values[item["relation"]]
            state = states[state_index]
            hook.set(state.a, state.b)
        reply = generate(model, tok, prompt, max_new_tokens)
        lp = gold_logprob(model, prompt, item["gold_id"])
        normalized = normalize_answer(reply)
        row = {
            "doc_index": source_doc,
            "relation": item["relation"],
            "gold": item["gold"],
            "reply": reply,
            "contains": normalize_answer(item["gold"]) in normalized,
            "exact": normalize_answer(item["gold"]) == normalized,
            "gold_logp": lp,
            "wrong_value": wrong_value,
            "wrong_value_contains": (
                wrong_value is not None
                and normalize_answer(wrong_value) in normalized
            ),
        }
        rows.append(row)
    hook.clear()
    return {
        "condition": condition,
        "n": len(rows),
        "contains": sum(int(row["contains"]) for row in rows),
        "exact": sum(int(row["exact"]) for row in rows),
        "wrong_value_contains": sum(int(row["wrong_value_contains"]) for row in rows),
        "mean_gold_logp": sum(row["gold_logp"] for row in rows) / len(rows),
        "rows": rows,
    }


@torch.inference_mode()
def dev_score(model, tok, compiler: ContextWeightCompiler,
              hook: DynamicLowRankHook, specs: list[EpisodeSpec],
              sources: list[ContextSource], device: torch.device) -> dict:
    total_lp = 0.0
    correct = 0
    count = 0
    route_correct = 0
    for doc_index, spec in enumerate(specs):
        a, b, transformed_keys = compiler.factors(sources[doc_index], device)
        hook.set(a, b)
        for relation in RELATIONS:
            prompt = prompt_ids(tok, relation["eval_question"], device)
            logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float()
            target = answer_id(tok, spec.values[relation["id"]], device)
            total_lp += float(F.log_softmax(logits, dim=-1)[0, target])
            correct += int(int(logits.argmax(dim=-1)) == target)
            route = F.normalize(hook.last_input, dim=-1) @ F.normalize(
                transformed_keys, dim=-1
            ).T
            expected = relation_slot(spec, relation["id"])
            route_correct += int(int(route.argmax(dim=-1)) == expected)
            count += 1
    hook.clear()
    return {
        "first_token_accuracy": correct / count,
        "mean_gold_logp": total_lp / count,
        "route_accuracy": route_correct / count,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    for value in SEEN_VALUES + OOD_VALUES:
        answer_id(tok, value, torch.device("cpu"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.device
    if args.layer < 0 or args.layer >= len(model.model.layers):
        raise ValueError("invalid compiler layer")

    train_specs = build_specs(args.seed * 1000 + 1, args.train_docs, SEEN_VALUES)
    dev_specs = build_specs(args.seed * 1000 + 10001, args.dev_docs, SEEN_VALUES)
    test_seen_specs = build_specs(
        args.seed * 1000 + 20001, args.test_docs, SEEN_VALUES
    )
    test_ood_specs = build_specs(
        args.seed * 1000 + 30001, args.test_docs, OOD_VALUES
    )
    test_seen_wrong_specs = deranged_value_specs(test_seen_specs)
    test_ood_wrong_specs = deranged_value_specs(test_ood_specs)

    print("capturing train/dev raw-context sources")
    t0 = time.time()
    train_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in train_specs
    ]
    dev_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in dev_specs
    ]
    print(f"captured {len(train_sources) + len(dev_sources)} documents in "
          f"{time.time() - t0:.1f}s")

    compiler = ContextWeightCompiler(
        hidden_size=int(model.config.hidden_size),
        intermediate_size=int(model.config.intermediate_size),
        key_rank=args.key_rank,
        residual_rank=args.value_rank,
        initial_gain=args.initial_gain,
    ).to(device)
    hook = DynamicLowRankHook(model, args.layer)
    optimizer = torch.optim.AdamW(
        compiler.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    examples = train_examples(tok, train_specs, device)
    rng = random.Random(args.seed + 41)
    marks = []
    best_state = copy.deepcopy(compiler.state_dict())
    best_dev = -float("inf")

    print("training shared context-to-weight compiler")
    train_start = time.time()
    for step in range(1, args.steps + 1):
        example = examples[rng.randrange(len(examples))]
        source = train_sources[example["doc_index"]]
        a, b, transformed_keys = compiler.factors(source, device)
        hook.set(a, b)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=example["prompt_ids"], use_cache=False
        ).logits[:, -1].float()
        target = torch.tensor([example["answer_id"]], device=device)
        answer_loss = F.cross_entropy(logits, target)
        if hook.last_input is None:
            raise RuntimeError("reader hook did not capture its input")
        route_logits = args.route_temperature * (
            F.normalize(hook.last_input, dim=-1)
            @ F.normalize(transformed_keys, dim=-1).T
        )
        expected_slot = relation_slot(
            train_specs[example["doc_index"]], example["relation"]
        )
        route_target = torch.tensor([expected_slot], device=device)
        route_loss = F.cross_entropy(route_logits, route_target)
        regularizer = (
            1e-5 * compiler.key_log_diagonal.square().mean()
            + 1e-8 * compiler.key_left.square().mean()
            + 1e-8 * compiler.key_right.square().mean()
        )
        loss = answer_loss + args.route_weight * route_loss + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compiler.parameters(), 1.0)
        optimizer.step()
        hook.clear()

        if step % args.eval_every == 0 or step == args.steps:
            compiler.eval()
            score = dev_score(
                model, tok, compiler, hook, dev_specs, dev_sources, device
            )
            compiler.train()
            mark = {
                "step": step,
                "train_answer_loss": float(answer_loss.detach()),
                "train_route_loss": float(route_loss.detach()),
                "gain": float(torch.exp(compiler.log_gain.detach())),
                **score,
            }
            marks.append(mark)
            print(
                f"step={step:4d} answer_loss={mark['train_answer_loss']:.3f} "
                f"route_loss={mark['train_route_loss']:.3f} "
                f"dev_logp={score['mean_gold_logp']:+.3f} "
                f"dev_top1={score['first_token_accuracy']:.3f} "
                f"route={score['route_accuracy']:.3f} gain={mark['gain']:.1f}"
            )
            if score["mean_gold_logp"] > best_dev:
                best_dev = score["mean_gold_logp"]
                best_state = copy.deepcopy(compiler.state_dict())

    compiler.load_state_dict(best_state)
    compiler.eval()
    training_seconds = time.time() - train_start

    # Target phase begins here.  Only raw target documents are opened by the
    # writer.  Evaluation items do not exist yet.
    print("compiling unseen target documents from raw context only")
    target_start = time.time()
    test_seen_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in test_seen_specs
    ]
    test_ood_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in test_ood_specs
    ]
    test_seen_wrong_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in test_seen_wrong_specs
    ]
    test_ood_wrong_sources = [
        capture_context_source(
            model, tok, spec.document, args.layer, device,
            value_source=args.value_source
        )
        for spec in test_ood_wrong_specs
    ]
    test_seen_states = [persist(compiler, source, device) for source in test_seen_sources]
    test_ood_states = [persist(compiler, source, device) for source in test_ood_sources]
    test_seen_wrong_states = [
        persist(compiler, source, device) for source in test_seen_wrong_sources
    ]
    test_ood_wrong_states = [
        persist(compiler, source, device) for source in test_ood_wrong_sources
    ]
    random_seen_states = [
        random_like(state, args.seed + 50000 + index)
        for index, state in enumerate(test_seen_states)
    ]
    random_ood_states = [
        random_like(state, args.seed + 60000 + index)
        for index, state in enumerate(test_ood_states)
    ]
    target_compile_seconds = time.time() - target_start

    # The blind-within-process read artifacts are instantiated only after all
    # target factors have been detached and converted to BF16.
    test_seen_items = make_eval_items(tok, test_seen_specs, device)
    test_ood_items = make_eval_items(tok, test_ood_specs, device)

    evaluations = {}
    for split_name, specs, wrong_specs, items, states, wrong_states, random_states in (
        ("seen_recombination", test_seen_specs, test_seen_wrong_specs,
         test_seen_items, test_seen_states, test_seen_wrong_states,
         random_seen_states),
        ("unseen_values", test_ood_specs, test_ood_wrong_specs,
         test_ood_items, test_ood_states, test_ood_wrong_states,
         random_ood_states),
    ):
        evaluations[split_name] = {
            "off": evaluate_condition(
                model, tok, hook, items, specs, None,
                args.max_new_tokens, "off"
            ),
            "full_context_oracle": evaluate_condition(
                model, tok, hook, items, specs, None,
                args.max_new_tokens, "oracle"
            ),
            "matched_compiled_weights": evaluate_condition(
                model, tok, hook, items, specs, states,
                args.max_new_tokens, "matched"
            ),
            "wrong_document_weights": evaluate_condition(
                model, tok, hook, items, specs, wrong_states,
                args.max_new_tokens, "wrong_document",
                wrong_specs=wrong_specs,
            ),
            "random_same_rank_norm": evaluate_condition(
                model, tok, hook, items, specs, random_states,
                args.max_new_tokens, "random"
            ),
        }
        base_lp = evaluations[split_name]["off"]["mean_gold_logp"]
        for result in evaluations[split_name].values():
            result["delta_mean_gold_logp_vs_off"] = (
                result["mean_gold_logp"] - base_lp
            )
        matched = evaluations[split_name]["matched_compiled_weights"]
        wrong_result = evaluations[split_name]["wrong_document_weights"]
        print(
            f"{split_name}: matched={matched['contains']}/{matched['n']} "
            f"wrong={wrong_result['contains']}/{wrong_result['n']} "
            f"wrong-value={wrong_result['wrong_value_contains']}/{wrong_result['n']}"
        )

    slots = len(RELATIONS)
    state_bytes = 2 * slots * (
        int(model.config.hidden_size) + int(model.config.intermediate_size)
    )
    report = {
        "method": "episodic_raw_context_to_low_rank_mlp_compiler",
        "status": "controlled_development_proof",
        "mainline_contract": {
            "outer_training": "separate synthetic episodes with QA loss",
            "target_writer_input": "raw target document only",
            "target_writer_calls": "one frozen context forward plus shared compiler",
            "target_forbidden_inputs": [
                "target questions", "target answers", "target clozes",
                "target fact table", "retrieval state",
            ],
            "persistent_state": "BF16 low-rank MLP down-projection factors",
            "reader_input": "question only; no document, tail, KV cache, or index",
        },
        "controlled_scope": {
            "document_grammar": "four shuffled key/value sentences",
            "value_width": "one tokenizer token",
            "rank": slots,
            "claim_excluded": "free-form document memory",
        },
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "train_docs": args.train_docs,
        "dev_docs": args.dev_docs,
        "test_docs_per_split": args.test_docs,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "route_weight": args.route_weight,
        "route_temperature": args.route_temperature,
        "key_residual_rank": args.key_rank,
        "value_residual_rank": args.value_rank,
        "value_source": args.value_source,
        "state_rank": slots,
        "state_bytes_per_document_bf16": state_bytes,
        "state_mebibytes_per_document_bf16": state_bytes / (1024 ** 2),
        "training_seconds": training_seconds,
        "target_compile_seconds": target_compile_seconds,
        "marks": marks,
        "best_dev_mean_gold_logp": best_dev,
        "compiler_parameter_count": sum(p.numel() for p in compiler.parameters()),
        "test_document_hashes": {
            "seen_recombination": [sha256_text(spec.document) for spec in test_seen_specs],
            "unseen_values": [sha256_text(spec.document) for spec in test_ood_specs],
        },
        "wrong_document_hashes": {
            "seen_recombination": [
                sha256_text(spec.document) for spec in test_seen_wrong_specs
            ],
            "unseen_values": [sha256_text(spec.document) for spec in test_ood_wrong_specs],
        },
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - t0,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    hook.close()
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
