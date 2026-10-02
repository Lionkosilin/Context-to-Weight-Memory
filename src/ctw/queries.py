"""Self-queries: the writer's own sample of the questions a document will be asked.

A write sees only the document C. The frozen model reads C and writes questions about it, and every
sentence also yields a cloze query. The teacher is the same frozen model with C in the prompt; its
greedy answer is appended to both sequences, so the student (question only) and the teacher share
the question, the template tail, and the answer at the end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .writers.base import Context

QUESTION_SYSTEM = "You write questions that test whether a reader remembers a text."
QUESTION_REQUEST = ("Write {n} short questions about the facts in the record above. Cover every fact, "
                    "and ask about each fact in more than one way. Each answer must be a few words "
                    "from the record. Write one question per line, without numbers or answers.")
CLOZE = 'Complete this sentence from the record: "{prefix} ..."'

_ITEM = re.compile(r"^\s*(?:[-*•]+|\(?\d+\s*[.):]|Q\d*\s*[.:)])\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Query:
    question: str
    source: str              # generated | cloze
    student: torch.Tensor    # 1 x n_s: read-time prompt, then the teacher's answer
    teacher: torch.Tensor    # 1 x n_t: prompt with the document, then the same answer
    shared: int              # trailing positions the two sequences share
    answer: int              # answer tokens at the end of both

    @property
    def head(self) -> int:
        """Leading student positions before the shared part: the same for every query."""
        return self.student.shape[1] - self.shared

    @property
    def student_shared(self) -> slice:
        return slice(self.head, self.student.shape[1])

    @property
    def teacher_shared(self) -> slice:
        n = self.teacher.shape[1]
        return slice(n - self.shared, n)

    @property
    def student_answer(self) -> slice:
        """Positions whose next-token logits predict the answer tokens."""
        n = self.student.shape[1]
        return slice(n - self.answer - 1, n - 1)

    @property
    def teacher_answer(self) -> slice:
        n = self.teacher.shape[1]
        return slice(n - self.answer - 1, n - 1)


def parse_questions(text: str, limit: int) -> list[str]:
    out, seen = [], set()
    for line in text.splitlines():
        q = _ITEM.sub("", line).strip().strip('"').strip()
        if not q.endswith("?") or len(q.split()) < 3 or len(q) > 200 or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
        if len(out) == limit:
            break
    return out


def cloze_questions(document: str) -> list[str]:
    out = []
    for sentence in _SENTENCE.split(document.strip()):
        words = sentence.strip().rstrip(".!?").split()
        if len(words) >= 4:
            out.append(CLOZE.format(prefix=" ".join(words[:-1])))
    return out


@torch.no_grad()
def greedy(ctx: Context, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    pad = ctx.tok.pad_token_id if ctx.tok.pad_token_id is not None else ctx.tok.eos_token_id
    out = ctx.model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                             max_new_tokens=max_new_tokens, pad_token_id=pad)
    return out[:, prompt.shape[1]:]


def generate_questions(ctx: Context, document: str, n: int) -> list[str]:
    user = f"{ctx.prompt.context_label}\n{document}\n\n{QUESTION_REQUEST.format(n=n)}"
    prompt = ctx.prompt.render(ctx.tok, user, QUESTION_SYSTEM, ctx.device)
    reply = ctx.tok.decode(greedy(ctx, prompt, 32 * n)[0], skip_special_tokens=True)
    return parse_questions(reply, n)


def _common_suffix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def synthesize(ctx: Context, document: str, questions: int, cloze: bool,
               answer_tokens: int) -> list[Query]:
    """Self-queries for one document, each paired with the teacher's answer."""
    ctx.hooks.set(None)
    texts = [(q, "generated") for q in (generate_questions(ctx, document, questions) if questions else [])]
    texts += [(q, "cloze") for q in (cloze_questions(document) if cloze else [])]
    out, seen = [], set()
    for text, source in texts:
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        student = ctx.prompt.ids(ctx.tok, text, device=ctx.device)
        teacher = ctx.prompt.ids(ctx.tok, text, context=document, device=ctx.device)
        common = _common_suffix(student[0].tolist(), teacher[0].tolist())
        answer = greedy(ctx, teacher, answer_tokens)
        if common == 0 or answer.shape[1] == 0:
            continue
        out.append(Query(text, source, torch.cat([student, answer], dim=1),
                         torch.cat([teacher, answer], dim=1), common + answer.shape[1], answer.shape[1]))
    if not out:
        raise ValueError("no usable self-queries: the document has no sentence of four or more words "
                         "and the model wrote no question")
    return out
