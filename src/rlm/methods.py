"""写入方法注册表。每个方法签名相同：(writer, tok, text, hp) -> info；只吃原始文本。

  ntp   下一词预测（TTT-E2E 的 naive 版；冻结骨干上已知无效，作对照）
  ttcd  末层隐藏状态 MSE：长窗教师 vs 短窗学生（arXiv:2608.01672 的冻结近似）
  dcd   深度上下文蒸馏（arXiv:2503.08727 文档版）：
        教师 = 冻结骨干看 [前文窗口 + 本块]，学生 = 带 ΔW 只看 [本块]；
        损失 = KL(教师 logits ‖ 学生 logits) + λ·Σ_l L1(隐藏状态)/‖教师隐藏‖₁
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .writer import DownWriter


def _chunks(n: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(n, s + chunk)) for s in range(0, n, chunk)]


def _encode(tok, text: str, device) -> torch.Tensor:
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def ntp(writer: DownWriter, tok, text: str, hp: dict) -> dict:
    ids = _encode(tok, text, writer.model.device)
    steps = sum(writer.write_document_ntp(ids, chunk=hp["chunk"]) for _ in range(hp["passes"]))
    return {"steps": steps}


def ttcd(writer: DownWriter, tok, text: str, hp: dict) -> dict:
    ids = _encode(tok, text, writer.model.device)
    steps = sum(writer.write_document(ids, chunk=hp["chunk"], prefix=hp["prefix"], recent=hp["recent"])
                for _ in range(hp["passes"]))
    return {"steps": steps}


def dcd(writer: DownWriter, tok, text: str, hp: dict) -> dict:
    model = writer.model
    ids = _encode(tok, text, model.device)
    n = ids.shape[1]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"])
    lam = hp["hidden_weight"]
    log = []
    step = 0
    for _ in range(hp["passes"]):
        for start, end in _chunks(n, hp["chunk"]):
            student = ids[:, start:end]
            teacher = ids[:, max(0, end - hp["recent"] - hp["chunk"]):end]
            m = student.shape[1]
            # 教师：冻结骨干（临时换回原始权重），看前文 + 本块
            with torch.no_grad():
                live = writer._snapshot_down()
                writer._copy_down(writer.originals)
                t = model(input_ids=teacher, output_hidden_states=True)
                t_logp = F.log_softmax(t.logits[:, -m:].float(), dim=-1)
                t_h = [h[:, -m:].float() for h in t.hidden_states[1:]]
                writer._copy_down(live)
                del t
            # 学生：带 ΔW，只看本块
            s = model(input_ids=student, output_hidden_states=True)
            s_logp = F.log_softmax(s.logits.float(), dim=-1)
            kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="none").sum(-1).mean()
            l1 = sum((sh.float() - th).abs().sum() / th.abs().sum()
                     for sh, th in zip(s.hidden_states[1:], t_h)) / len(t_h)
            loss = kl + lam * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                log.append({"step": step, "kl": kl.item(), "l1": l1.item()})
                print(f"  step {step} kl={kl.item():.3f} l1={l1.item():.4f}", flush=True)
    return {"steps": step, "log": log}


METHODS = {"ntp": ntp, "ttcd": ttcd, "dcd": dcd}
