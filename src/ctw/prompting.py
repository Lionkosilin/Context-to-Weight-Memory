"""Turn a question (and optionally a document) into prompt token ids."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PromptFormat:
    system: str = "Answer with only the exact requested value, without explanation."
    chat_template: str | bool = "auto"   # "auto" uses the tokenizer's template when it has one
    context_label: str = "Reference record:"
    question_label: str = "Question:"
    answer_label: str = "Answer:"        # plain-text format only
    enable_thinking: bool = False        # passed to templates that read it (Qwen3)

    def uses_chat(self, tok) -> bool:
        if self.chat_template == "auto":
            return bool(getattr(tok, "chat_template", None))
        return bool(self.chat_template)

    def ids(self, tok, question: str, context: str | None = None,
            device: torch.device | str = "cpu") -> torch.Tensor:
        user = f"{self.context_label}\n{context}\n\n" if context is not None else ""
        user += f"{self.question_label} {question}"
        return self.render(tok, user, self.system, device)

    def render(self, tok, user: str, system: str | None,
               device: torch.device | str = "cpu") -> torch.Tensor:
        """One user turn, ready for the reply, with an optional system message."""
        if self.uses_chat(tok):
            messages = [{"role": "user", "content": user}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            enc = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=self.enable_thinking, return_tensors="pt",
            )
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        else:
            text = (f"{system}\n\n" if system else "") + f"{user}\n{self.answer_label}"
            ids = tok(text, return_tensors="pt").input_ids
        return ids.to(device)


def encode(tok, text: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """A standalone text as the model reads it, with the tokenizer's start token if it has one."""
    return tok(text, return_tensors="pt").input_ids.to(device)


def answer_ids(tok, answer: str, chat: bool) -> list[int]:
    """Token ids of the answer as the model would emit it right after the prompt."""
    return tok(answer if chat else " " + answer, add_special_tokens=False).input_ids
