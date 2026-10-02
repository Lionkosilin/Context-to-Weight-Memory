"""Writer interface and registry. A new algorithm is one subclass with @register."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from ..adapter import ModelAdapter
from ..memory import MemoryHooks, MemoryState
from ..prompting import PromptFormat
from ..stats import StatsCache
from ..tasks import Episode

WRITERS: dict[str, type[Writer]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        WRITERS[name] = cls
        return cls
    return deco


def build_writer(name: str, params: dict | None = None) -> Writer:
    if name not in WRITERS:
        raise ValueError(f"unknown writer {name!r}; available: {sorted(WRITERS)}")
    return WRITERS[name](params or {})


@dataclass
class Context:
    """Everything a writer may touch. Questions are not part of it."""

    model: torch.nn.Module
    tok: object
    adapter: ModelAdapter
    hooks: MemoryHooks
    layers: list[int]
    prompt: PromptFormat
    seed: int = 0
    stats: StatsCache | None = None   # key second moments on generic text, per layer

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def chat(self) -> bool:
        return self.prompt.uses_chat(self.tok)


class Writer:
    """write(ctx, document) -> MemoryState is the whole contract.

    Writers that learn across documents also implement fit(); fit sees training episodes
    (with their questions), never the evaluation episodes.
    """

    name: ClassVar[str] = ""
    trainable: ClassVar[bool] = False

    @dataclass
    class Params:
        pass

    def __init__(self, params: dict):
        fields = {f.name for f in dataclasses.fields(self.Params)}
        unknown = set(params) - fields
        if unknown:
            raise ValueError(f"writer {self.name!r}: unknown params {sorted(unknown)}; "
                             f"accepted: {sorted(fields)}")
        self.p = self.Params(**params)

    def fit(self, ctx: Context, train: list[Episode], dev: list[Episode]) -> dict:
        return {}

    def write(self, ctx: Context, document: str) -> MemoryState:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Persist what fit() learned."""

    def load(self, path: str | Path, ctx: Context) -> None:
        """Restore what fit() learned."""
