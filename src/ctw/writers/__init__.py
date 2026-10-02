import importlib
import pkgutil

from .base import WRITERS, Context, Writer, build_writer, register

# Every module in this package registers its writers on import.
for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")

__all__ = ["WRITERS", "Context", "Writer", "build_writer", "register"]
