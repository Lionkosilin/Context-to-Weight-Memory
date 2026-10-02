"""Tasks: documents plus the questions asked after the document is written.

A task builds named splits of episodes. `fit_splits` feed writers that learn across documents
(ACWC). `eval_splits` are written first and questioned afterwards.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

TASKS: dict[str, Callable[..., "Task"]] = {}


def register_task(name: str):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


def build_task(name: str, params: dict, tok, seed: int) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name](tok=tok, seed=seed, **params)


@dataclass
class Question:
    id: str
    text: str
    gold: str
    aliases: list[str] = field(default_factory=list)
    wrong_gold: str | None = None   # the answer under the decoy document
    meta: dict = field(default_factory=dict)


@dataclass
class Episode:
    id: str
    document: str
    questions: list[Question]
    decoy: str | None = None                 # document for the wrong-document control
    train_questions: list[Question] = field(default_factory=list)


@dataclass
class Task:
    name: str
    splits: dict[str, list[Episode]]
    fit_splits: tuple[str, str] | None = None   # (train, dev)
    eval_splits: tuple[str, ...] = ()


# ---------------------------------------------------------------- synthetic key/value

RELATIONS = ("navigation", "emergency", "archive", "calibration")
STATEMENT = "The {relation} key was {value}."
TRAIN_QUESTIONS = (
    "What was the {relation} key?",
    "Which word was assigned as the {relation} key?",
    "State the {relation} key.",
)
EVAL_QUESTION = "Report the assigned value of the {relation} key."
SEEN_VALUES = (
    "amber", "silver", "golden", "blue", "green", "orange", "purple", "black",
    "white", "yellow", "red", "brown", "pink", "gray", "rabbit", "lion",
)
UNSEEN_VALUES = ("wolf", "bear", "horse", "fox", "owl", "mouse", "cat", "dog")


def _single_token(tok, word: str) -> bool:
    """One token as a chat reply ("word") or as a plain-text continuation (" word")."""
    return any(len(tok(w, add_special_tokens=False).input_ids) == 1 for w in (word, " " + word))


def _kv_episode(seed: int, relations, pool) -> tuple[dict[str, str], list[str]]:
    rng = random.Random(seed)
    values = dict(zip(relations, rng.sample(list(pool), k=len(relations))))
    order = list(relations)
    rng.shuffle(order)
    return values, order


def _kv_document(values: dict[str, str], order: list[str]) -> str:
    return " ".join(STATEMENT.format(relation=r, value=values[r]) for r in order)


def _kv_split(base: int, count: int, relations, pool, name: str) -> list[Episode]:
    episodes, seen, cursor = [], set(), base
    while len(episodes) < count:
        values, order = _kv_episode(cursor, relations, pool)
        cursor += 1
        signature = tuple(values[r] for r in relations)
        if signature in seen:
            continue
        seen.add(signature)
        rotated = dict(zip(relations, [values[r] for r in relations][1:] + [values[relations[0]]]))
        eid = f"{name}-{len(episodes)}"
        episodes.append(Episode(
            id=eid,
            document=_kv_document(values, order),
            decoy=_kv_document(rotated, order),
            questions=[Question(f"{eid}-{r}", EVAL_QUESTION.format(relation=r), values[r],
                                wrong_gold=rotated[r], meta={"slot": order.index(r)})
                       for r in relations],
            train_questions=[Question(f"{eid}-{r}-t{k}", q.format(relation=r), values[r],
                                      meta={"slot": order.index(r)})
                             for r in relations for k, q in enumerate(TRAIN_QUESTIONS)],
        ))
    return episodes


@register_task("synthetic_kv")
def synthetic_kv(tok, seed: int, train_docs: int = 64, dev_docs: int = 8, test_docs: int = 8,
                 relations=RELATIONS, seen_values=SEEN_VALUES, unseen_values=UNSEEN_VALUES) -> Task:
    """Four shuffled "The <relation> key was <value>." sentences per document.

    seen_recombination: values seen in fit, new combinations.
    unseen_values: values never seen in fit.
    Every value must be one token for the loaded tokenizer; override the value lists otherwise.
    """
    relations = tuple(relations)
    bad = [v for v in (*seen_values, *unseen_values) if not _single_token(tok, v)]
    if bad:
        raise ValueError(f"values are not single tokens for this tokenizer: {bad}. "
                         "Set task.params.seen_values / unseen_values to single-token words.")
    if len(seen_values) < len(relations) or len(unseen_values) < len(relations):
        raise ValueError("each value pool needs at least one value per relation")
    return Task(
        name="synthetic_kv",
        splits={
            "train": _kv_split(seed * 1000 + 1, train_docs, relations, seen_values, "train"),
            "dev": _kv_split(seed * 1000 + 10001, dev_docs, relations, seen_values, "dev"),
            "seen_recombination": _kv_split(seed * 1000 + 20001, test_docs, relations, seen_values, "seen"),
            "unseen_values": _kv_split(seed * 1000 + 30001, test_docs, relations, unseen_values, "unseen"),
        },
        fit_splits=("train", "dev"),
        eval_splits=("seen_recombination", "unseen_values"),
    )


# ---------------------------------------------------------------- memo fixture

MEMO = (
    ("navigator badge", "The navigator badge was {}.", "amber", "violet",
     "What color was the navigator badge in the Aster Vale memorandum?"),
    ("signal phrase", "The emergency signal phrase was {}.", "silent meadow", "quiet harbor",
     "What was the emergency signal phrase in the Aster Vale memorandum?"),
    ("crate code", "The archive crate code was {}.", "Q7-LUMEN", "M3-CEDAR",
     "What was the archive crate code in the Aster Vale memorandum?"),
    ("pressure", "The west chamber pressure target was {}.", "37 kilopascals", "52 kilopascals",
     "What was the west chamber pressure target in the Aster Vale memorandum?"),
    ("storage", "The mineral sample was stored in {}.", "bay 4", "bay 8",
     "Where was the mineral sample stored in the Aster Vale memorandum?"),
    ("day", "The morning calibration began on {}.", "Tuesday", "Friday",
     "On which day did the morning calibration begin in the Aster Vale memorandum?"),
)


@register_task("memo")
def memo(tok, seed: int) -> Task:
    """One memorandum with six facts (names, codes, quantities, days) and a matched decoy."""
    head = "Aster Vale Observatory issued a sealed calibration memorandum. "
    doc = head + " ".join(s.format(g) for _, s, g, _, _ in MEMO)
    decoy = head + " ".join(s.format(d) for _, s, _, d, _ in MEMO)
    qs = [Question(f"memo-{i}", q, g, wrong_gold=d) for i, (_, _, g, d, q) in enumerate(MEMO)]
    return Task("memo", {"memo": [Episode("memo", doc, qs, decoy=decoy)]}, eval_splits=("memo",))


# ---------------------------------------------------------------- JSON question bank

@register_task("qa_json")
def qa_json(tok, seed: int, path: str, split: str | None = None) -> Task:
    """Your own documents and questions.

    {"episodes": [{"id": "doc1", "document": "..." | "document_path": "file.txt",
                   "decoy_document": "..." (optional),
                   "questions": [{"id": "q1", "question": "...", "gold": "...",
                                  "aliases": [], "wrong_gold": null, "split": "check"}]}]}

    `split` keeps only questions with that split label.
    """
    root = Path(path).parent
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = []
    for e in data["episodes"]:
        doc = e.get("document")
        if doc is None:
            doc = (root / e["document_path"]).read_text(encoding="utf-8")
        qs = [Question(q["id"], q["question"], q["gold"], q.get("aliases", []), q.get("wrong_gold"))
              for q in e["questions"] if split is None or q.get("split") == split]
        if qs:
            episodes.append(Episode(e["id"], doc, qs, decoy=e.get("decoy_document")))
    if not episodes:
        raise ValueError(f"{path}: no questions left for split={split!r}")
    return Task("qa_json", {"qa": episodes}, eval_splits=("qa",))
