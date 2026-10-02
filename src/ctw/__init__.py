"""Context-to-Weight Memory: write a document into ΔW on a frozen LM, read it back without the document."""

from .adapter import ModelAdapter, resolve_layers
from .memory import DenseDelta, LowRankDelta, MemoryHooks, MemoryState
from .prompting import PromptFormat
from .tasks import Episode, Question, Task, register_task
from .writers import Context, Writer, build_writer, register

__version__ = "0.2.0"

__all__ = [
    "Context", "DenseDelta", "Episode", "LowRankDelta", "MemoryHooks", "MemoryState",
    "ModelAdapter", "PromptFormat", "Question", "Task", "Writer", "build_writer",
    "register", "register_task", "resolve_layers",
]
