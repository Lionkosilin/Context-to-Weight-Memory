# Inference-time write: backbone frozen. Only down_proj delta gets SGD.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _layer_list(model: nn.Module):
    return model.model.layers


def _down(layer) -> nn.Parameter:
    return layer.mlp.down_proj.weight


def select_down_layers(n_total: int, n: int, pick: str) -> list[int]:
    """Which mlp.down_proj tensors to write. `n` is the bit-budget count."""
    if n <= 0 or n > n_total:
        raise ValueError(f"n={n} not in 1..{n_total}")
    if pick == "last":
        return list(range(n_total - n, n_total))
    if pick == "first":
        return list(range(n))
    if pick in ("mid", "mid25_50"):
        lo = int(round(0.25 * n_total))
        hi = min(n_total - 1, int(round(0.50 * n_total)))
        if n == 1:
            return [lo]
        raw = [int(round(lo + i * (hi - lo) / (n - 1))) for i in range(n)]
        out: list[int] = []
        for x in raw:
            x = min(max(x, lo), hi)
            if x not in out:
                out.append(x)
        for x in range(lo, hi + 1):
            if len(out) >= n:
                break
            if x not in out:
                out.append(x)
        return sorted(out)
    raise ValueError(f"unknown pick {pick!r}")


@dataclass
class DownWriter:
    """Per-document ΔW on selected mlp.down_proj.weight. Call zero_() between docs."""

    model: nn.Module
    layer_ids: list[int]
    lr: float
    originals: dict[int, torch.Tensor]
    opt: torch.optim.Optimizer

    @classmethod
    def attach(
        cls,
        model: nn.Module,
        n_layers: int = 8,
        lr: float = 1e-4,
        pick: str = "last",
        layer_ids: list[int] | None = None,
    ) -> DownWriter:
        model.eval()
        layers = _layer_list(model)
        ids = (
            list(layer_ids)
            if layer_ids is not None
            else select_down_layers(len(layers), n_layers, pick)
        )
        if not ids:
            raise ValueError("no layers to write")
        for p in model.parameters():
            p.requires_grad_(False)
        originals = {}
        delta_params = []
        for i in ids:
            w = _down(layers[i])
            w.requires_grad_(True)
            originals[i] = w.detach().clone()
            delta_params.append(w)
        opt = torch.optim.SGD(delta_params, lr=lr)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == len(ids)
        return cls(model, ids, lr, originals, opt)

    def delta(self) -> dict[str, torch.Tensor]:
        layers = _layer_list(self.model)
        return {
            f"layers.{i}.mlp.down_proj.weight": _down(layers[i]).detach() - self.originals[i]
            for i in self.layer_ids
        }

    def delta_norm(self) -> float:
        return float(sum(v.float().norm() ** 2 for v in self.delta().values()).sqrt())

    def weight_rel_norm(self) -> float:
        """||ΔW||_F / ||W0||_F over written tensors."""
        layers = _layer_list(self.model)
        num = 0.0
        den = 0.0
        for i in self.layer_ids:
            d = (_down(layers[i]).detach() - self.originals[i]).float()
            w0 = self.originals[i].float()
            num += float(d.norm() ** 2)
            den += float(w0.norm() ** 2)
        if den <= 0:
            return 0.0
        return (num / den) ** 0.5

    def zero_(self) -> None:
        """Clear ΔW before the next document. Same as restore."""
        self.restore()

    def restore(self) -> None:
        layers = _layer_list(self.model)
        with torch.no_grad():
            for i, w0 in self.originals.items():
                _down(layers[i]).copy_(w0)
        self.opt = torch.optim.SGD(
            [_down(_layer_list(self.model)[i]) for i in self.layer_ids], lr=self.lr
        )

    def n_delta_tensors(self) -> int:
        return sum(p.requires_grad for p in self.model.parameters())

    def write_document(
        self,
        ids: torch.Tensor,
        chunk: int = 512,
        prefix: int = 512,
        recent: int = 2048,
    ) -> int:
        """IP-TTCD over chunks after `prefix` (prefix is teacher-only)."""
        n = ids.shape[1]
        if prefix >= n or chunk <= 0:
            return 0
        steps = 0
        start = prefix
        while start < n:
            end = min(n, start + chunk)
            student = ids[:, start:end]
            if student.shape[1] == 0:
                break
            teacher_start = 0 if end <= prefix + recent else end - recent
            prefix_ids = ids[:, :prefix]
            recent_ids = ids[:, max(prefix, teacher_start) : end]
            if recent_ids.shape[1] == 0:
                start = end
                continue
            teacher = torch.cat([prefix_ids, recent_ids], dim=1)
            if teacher.shape[1] == student.shape[1]:
                start = end
                continue
            self.step_ttcd(teacher, student)
            steps += 1
            start = end
        return steps

    def write_document_ntp(self, ids: torch.Tensor, chunk: int = 512) -> int:
        """Next-token write on every chunk, including the first."""
        n = ids.shape[1]
        steps = 0
        ends = list(range(chunk, n + 1, chunk))
        if n > 0 and (not ends or ends[-1] != n):
            ends.append(n)
        for end in ends:
            start = max(0, (end - 1) // chunk * chunk)
            student = ids[:, start:end]
            if student.shape[1] == 0:
                continue
            self.step_ntp(student)
            steps += 1
        return steps

    def step_ntp(self, input_ids: torch.Tensor) -> float:
        self.opt.zero_grad(set_to_none=True)
        out = self.model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        self.opt.step()
        return float(out.loss.detach())

    def step_ttcd(self, teacher_ids: torch.Tensor, student_ids: torch.Tensor) -> float:
        n = student_ids.shape[1]
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            th = self.model(input_ids=teacher_ids, output_hidden_states=True)
            teacher_h = th.hidden_states[-1][:, -n:, :].float()
        sh = self.model(input_ids=student_ids, output_hidden_states=True)
        student_h = sh.hidden_states[-1].float()
        loss = F.mse_loss(student_h, teacher_h)
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    def _copy_down(self, src: dict[int, torch.Tensor]) -> None:
        layers = _layer_list(self.model)
        with torch.no_grad():
            for i, w0 in src.items():
                _down(layers[i]).copy_(w0)

    def _snapshot_down(self) -> dict[int, torch.Tensor]:
        layers = _layer_list(self.model)
        return {i: _down(layers[i]).detach().clone() for i in self.layer_ids}

    def ce_nll(self, input_ids: torch.Tensor, n_prompt: int) -> torch.Tensor:
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100
        return self.model(input_ids=input_ids, labels=labels).loss

    @torch.no_grad()
    def frozen_last_logits(self, ctrl_ids: torch.Tensor) -> torch.Tensor:
        """Last-position logits under original W_down. Safe only with no live graph."""
        saved = self._snapshot_down()
        self._copy_down(self.originals)
        logits0 = self.model(input_ids=ctrl_ids).logits.float()[:, -1].clone()
        self._copy_down(saved)
        return logits0

    def kl_keep(
        self, ctrl_ids: torch.Tensor, frozen_last: torch.Tensor
    ) -> torch.Tensor:
        """KL(student || frozen) on the last position. No weight swapping."""
        logits1 = self.model(input_ids=ctrl_ids).logits.float()[:, -1]
        log_p0 = F.log_softmax(frozen_last.detach(), dim=-1)
        p1 = F.softmax(logits1, dim=-1)
        return F.kl_div(log_p0, p1, reduction="batchmean")

    def step_chunk(
        self,
        ids: torch.Tensor,
        start: int,
        end: int,
        packed_facts: list[tuple[torch.Tensor, int]],
        ctrl_ids: torch.Tensor | None = None,
        frozen_ctrl_logits: torch.Tensor | None = None,
        prefix: int = 512,
        recent: int = 2048,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.3,
        lam: float = 0.5,
    ) -> dict[str, float]:
        """One SGD step: CE-mem + NTP + TTCD + KL-keep. Single backward."""
        student = ids[:, start:end]
        if student.shape[1] == 0:
            return {}
        self.opt.zero_grad(set_to_none=True)
        terms: dict[str, float] = {}
        total = 0.0
        stepped = False

        def backward_term(name: str, weight: float, term: torch.Tensor) -> None:
            nonlocal total, stepped
            raw = float(term.detach())
            terms[name] = raw
            total += weight * raw
            (weight * term).backward()
            stepped = True

        teacher = None
        if end > prefix and student.shape[1] > 0 and gamma != 0:
            teacher_start = 0 if end <= prefix + recent else end - recent
            recent_ids = ids[:, max(prefix, teacher_start) : end]
            if recent_ids.shape[1] > 0:
                teacher = torch.cat([ids[:, :prefix], recent_ids], dim=1)
                if teacher.shape[1] == student.shape[1]:
                    teacher = None

        need_student = beta != 0 or teacher is not None
        if need_student:
            out = self.model(
                input_ids=student,
                labels=student if beta != 0 else None,
                output_hidden_states=teacher is not None,
            )
            chunk_loss = None
            if beta != 0:
                terms["ntp"] = float(out.loss.detach())
                total += beta * terms["ntp"]
                chunk_loss = beta * out.loss
            if teacher is not None:
                n = student.shape[1]
                with torch.no_grad():
                    th = self.model(input_ids=teacher, output_hidden_states=True)
                    teacher_h = th.hidden_states[-1][:, -n:, :].float()
                ttcd = F.mse_loss(out.hidden_states[-1].float(), teacher_h)
                terms["ttcd"] = float(ttcd.detach())
                total += gamma * terms["ttcd"]
                chunk_loss = ttcd * gamma if chunk_loss is None else chunk_loss + gamma * ttcd
            if chunk_loss is not None:
                chunk_loss.backward()
                stepped = True

        if packed_facts and alpha != 0:
            mem_sum = 0.0
            scale = alpha / len(packed_facts)
            for full, n_prompt in packed_facts:
                term = self.ce_nll(full, n_prompt)
                mem_sum += float(term.detach())
                (scale * term).backward()
                stepped = True
            terms["mem"] = mem_sum / len(packed_facts)
            total += alpha * terms["mem"]

        if ctrl_ids is not None and frozen_ctrl_logits is not None and lam != 0:
            backward_term("keep", lam, self.kl_keep(ctrl_ids, frozen_ctrl_logits))

        if not stepped:
            return terms
        terms["total"] = total
        self.opt.step()
        return terms

    def write_document_hybrid(
        self,
        ids: torch.Tensor,
        facts: list[tuple[str, str]],
        tok,
        ctrl_ids: torch.Tensor | None = None,
        frozen_ctrl_logits: torch.Tensor | None = None,
        chunk: int = 512,
        prefix: int = 512,
        recent: int = 2048,
        window: int = 512,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.3,
        lam: float = 0.5,
    ) -> list[dict[str, float]]:
        """Read the doc once. CE on document golds given the current last-`window`."""
        n = ids.shape[1]
        if n == 0 or chunk <= 0:
            return []
        seen: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        log: list[dict[str, float]] = []
        ends = list(range(chunk, n + 1, chunk))
        if not ends or ends[-1] != n:
            ends.append(n)
        for end in ends:
            start = max(0, (end - 1) // chunk * chunk)
            span = tok.decode(ids[0, start:end], skip_special_tokens=True)
            for cloze, gold in facts:
                key = (cloze, gold)
                if key not in seen_keys and gold in span:
                    seen.append(key)
                    seen_keys.add(key)
            packed: list[tuple[torch.Tensor, int]] = []
            if seen:
                tail = ids[:, max(0, end - window) : end]
                tail_text = tok.decode(tail[0], skip_special_tokens=True)
                for cloze, gold in seen:
                    prompt = tail_text + "\n" + cloze
                    full, n_prompt = pack_prompt_gold(tok, prompt, gold)
                    packed.append((full.to(ids.device), n_prompt))
            terms = self.step_chunk(
                ids,
                start,
                end,
                packed,
                ctrl_ids=ctrl_ids,
                frozen_ctrl_logits=frozen_ctrl_logits,
                prefix=prefix,
                recent=recent,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lam=lam,
            )
            if terms:
                terms["start"] = float(start)
                terms["end"] = float(end)
                terms["n_facts"] = float(len(seen))
                log.append(terms)
        return log


def pack_prompt_gold(tok, prompt: str, gold: str) -> tuple[torch.Tensor, int]:
    """ids = prompt + gold continuation; returns ids, n_prompt."""
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    full_ids = tok(prompt + " " + gold, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[1]
    if full_ids.shape[1] <= n_prompt:
        extra = tok(" " + gold, add_special_tokens=False, return_tensors="pt").input_ids
        full_ids = torch.cat([prompt_ids, extra], dim=1)
    return full_ids, n_prompt

