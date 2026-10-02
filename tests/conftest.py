"""Tiny random models and a word-level tokenizer, built offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ctw import queries, tasks
from ctw.config import DEFAULTS

ROOT = Path(__file__).resolve().parents[1]
CHAT = ("{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </turn> {% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}")


def _corpus() -> str:
    parts = [CHAT, DEFAULTS["prompt"]["system"], "Answer with only the assigned word, without explanation.",
             "Reference record: Question: Answer: </s> <unk> <pad>",
             queries.QUESTION_SYSTEM, queries.QUESTION_REQUEST, queries.CLOZE]
    for rel in tasks.RELATIONS:
        parts.append(tasks.STATEMENT.format(relation=rel, value=""))
        parts += [q.format(relation=rel) for q in (*tasks.TRAIN_QUESTIONS, tasks.EVAL_QUESTION)]
    parts += [*tasks.SEEN_VALUES, *tasks.UNSEEN_VALUES]
    parts += [" ".join(m[1:]) + " Aster Vale Observatory issued a sealed calibration memorandum." for m in tasks.MEMO]
    bank = json.loads((ROOT / "examples" / "lighthouse.json").read_text())
    for e in bank["episodes"]:
        parts += [e["document"], e["decoy_document"]]
        parts += [" ".join(str(v) for v in q.values()) for q in e["questions"]]
    return "\n".join(parts)


@pytest.fixture(scope="session")
def tok():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Whitespace()
    words = sorted({w for w, _ in pre.pre_tokenize_str(_corpus())})
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab.setdefault(w, len(vocab))
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre
    t = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    t.chat_template = CHAT
    return t


def tiny_config(arch: str, vocab: int):
    from transformers import GPT2Config, GPTNeoXConfig, LlamaConfig, OPTConfig, Qwen3Config

    common = dict(vocab_size=vocab, pad_token_id=1, eos_token_id=2, bos_token_id=2)
    if arch == "qwen3":
        return Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
                           max_position_embeddings=4096, **common)
    if arch == "llama":
        return LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                           num_key_value_heads=2, tie_word_embeddings=False,
                           max_position_embeddings=4096, **common)
    if arch == "gpt2":
        return GPT2Config(n_embd=32, n_layer=3, n_head=4, n_positions=4096, **common)
    if arch == "gpt_neox":
        return GPTNeoXConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             max_position_embeddings=4096, **common)
    if arch == "opt":
        return OPTConfig(hidden_size=32, ffn_dim=64, num_hidden_layers=2, num_attention_heads=4,
                         word_embed_proj_dim=32, max_position_embeddings=4096, **common)
    raise ValueError(arch)


ARCHS = ("qwen3", "llama", "gpt2", "gpt_neox", "opt")


@pytest.fixture(scope="session")
def model_dirs(tok, tmp_path_factory):
    """Save each tiny model with the tokenizer so load_model() reads it like a real checkpoint."""
    from transformers import AutoModelForCausalLM

    out = {}
    for arch in ARCHS:
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(tiny_config(arch, len(tok)))
        d = tmp_path_factory.mktemp(arch)
        model.save_pretrained(d)
        tok.save_pretrained(d)
        out[arch] = str(d)
    return out


def config_for(model_dir: str, tmp_path, **over) -> dict:
    from ctw.config import merge

    base = {"model": {"id": model_dir, "device": "cpu", "dtype": "float32"},
            "output": str(tmp_path / "{name}" / "seed{seed}.json"),
            "stats": {"tokens": 512, "chunk": 64, "batch": 4, "cache": str(tmp_path / "stats" / "{model}")},
            "eval": {"max_new_tokens": 2}}
    return merge(merge(DEFAULTS, base), over)
