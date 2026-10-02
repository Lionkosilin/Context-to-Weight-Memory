"""Find the decoder layers, the MLP output projection, and the output embeddings of any causal LM.

Paths are detected from a list of known layouts. Pass `layers_path` and `out_proj` to override
detection for an architecture that is not listed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

LAYER_PATHS = (
    "model.layers",                 # Llama, Qwen2/3, Mistral, Gemma, Phi-3
    "model.language_model.layers",  # multimodal wrappers
    "model.decoder.layers",         # OPT
    "transformer.h",                # GPT-2, GPT-J, Falcon, Bloom
    "gpt_neox.layers",              # GPT-NeoX, Pythia
    "transformer.blocks",           # MPT
)

OUT_PROJ_PATHS = (
    "mlp.down_proj",
    "mlp.c_proj",
    "mlp.dense_4h_to_h",
    "mlp.fc_out",
    "mlp.fc2",
    "fc2",
    "feed_forward.w2",
    "ffn.down_proj",
)


def get_path(root: nn.Module, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _try_path(root: nn.Module, path: str):
    try:
        return get_path(root, path)
    except (AttributeError, IndexError, TypeError):
        return None


def linear_dims(module: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) for nn.Linear and GPT-2 style Conv1D."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    weight = getattr(module, "weight", None)
    if type(module).__name__ == "Conv1D" and weight is not None:
        return int(weight.shape[0]), int(weight.shape[1])
    raise TypeError(f"cannot read input/output size of {type(module).__name__}")


@dataclass
class ModelAdapter:
    model: nn.Module
    layers: nn.ModuleList
    layers_path: str
    out_proj_path: str

    @classmethod
    def from_model(cls, model: nn.Module, layers_path: str | None = None,
                   out_proj: str | None = None) -> ModelAdapter:
        layers, lp = None, layers_path
        for path in ([layers_path] if layers_path else LAYER_PATHS):
            found = _try_path(model, path)
            if isinstance(found, (nn.ModuleList, nn.Sequential)) and len(found) > 0:
                layers, lp = found, path
                break
        if layers is None:
            raise ValueError(
                "decoder layers not found; set model.layers_path (e.g. 'model.layers'). "
                f"Tried: {', '.join(LAYER_PATHS)}"
            )
        op = None
        for path in ([out_proj] if out_proj else OUT_PROJ_PATHS):
            found = _try_path(layers[0], path)
            if isinstance(found, nn.Module) and hasattr(found, "weight"):
                op = path
                break
        if op is None:
            raise ValueError(
                "MLP output projection not found; set model.out_proj (e.g. 'mlp.down_proj'). "
                f"Tried: {', '.join(OUT_PROJ_PATHS)}"
            )
        return cls(model=model, layers=layers, layers_path=lp, out_proj_path=op)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def out_proj(self, layer: int) -> nn.Module:
        return get_path(self.layers[layer], self.out_proj_path)

    def dims(self, layer: int) -> tuple[int, int]:
        """(d_in, d_out) of the output projection: (d_ff, d_model) for a standard MLP."""
        return linear_dims(self.out_proj(layer))

    def weight(self, layer: int) -> torch.Tensor:
        """The output projection's weight in (d_out x d_in) orientation."""
        module = self.out_proj(layer)
        w = module.weight
        return w if isinstance(module, nn.Linear) else w.T

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def layer_output(self, output) -> torch.Tensor:
        """Hidden states from a decoder layer's forward output (a tensor or a tuple)."""
        return output[0] if isinstance(output, tuple) else output

    def output_embeddings(self) -> torch.Tensor:
        """Vocabulary x d_model matrix that maps hidden states to logits."""
        return self.model.get_output_embeddings().weight

    def resolve_layers(self, spec) -> list[int]:
        return resolve_layers(spec, self.n_layers)

    def describe(self) -> dict:
        d_in, d_out = self.dims(0)
        return {
            "model_type": getattr(self.model.config, "model_type", type(self.model).__name__),
            "n_layers": self.n_layers,
            "layers_path": self.layers_path,
            "out_proj": self.out_proj_path,
            "out_proj_type": type(self.out_proj(0)).__name__,
            "d_in": d_in,
            "d_out": d_out,
            "vocab": int(self.output_embeddings().shape[0]),
            "tied_embeddings": bool(getattr(self.model.config, "tie_word_embeddings", False)),
        }


def resolve_layers(spec, n_layers: int) -> list[int]:
    """Turn a layer spec into sorted, unique layer indices.

    Accepted forms, alone or in a list:
      7, -4            index; negatives count from the end
      0.89             fraction of depth (0 < f < 1)
      "last:8"         the last 8 layers
      "first:8"        the first 8 layers
      "all"            every layer
    """
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    out: set[int] = set()
    for item in items:
        if isinstance(item, str):
            if item == "all":
                out.update(range(n_layers))
                continue
            kind, _, count = item.partition(":")
            if kind in ("last", "first") and count.isdigit():
                k = int(count)
                if not 0 < k <= n_layers:
                    raise ValueError(f"{item!r} needs 1..{n_layers} layers")
                out.update(range(n_layers - k, n_layers) if kind == "last" else range(k))
                continue
            try:
                item = float(item) if "." in item else int(item)
            except ValueError:
                raise ValueError(f"unknown layer spec {item!r}") from None
        if isinstance(item, float):
            if not 0.0 < item < 1.0:
                raise ValueError(f"fractional layer {item} must lie in (0, 1)")
            out.add(min(n_layers - 1, round(item * n_layers)))
        elif isinstance(item, int):
            idx = item + n_layers if item < 0 else item
            if not 0 <= idx < n_layers:
                raise ValueError(f"layer {item} outside 0..{n_layers - 1}")
            out.add(idx)
        else:
            raise ValueError(f"unknown layer spec {item!r}")
    if not out:
        raise ValueError("layer spec selects no layers")
    return sorted(out)
