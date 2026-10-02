# Context-to-Weight Memory

[English](README.md) · [中文](README.zh-CN.md)

## 缘起

现有的上下文工程臃肿而复杂，我无法忍受。人的大脑能够记忆，并不依赖外部的库，而是靠独特的内化能力。既然模型在一定规模后会出现疑似涌现的能力，它也一定具备这种内化的潜力。这个课题由此开始。

冻结的模型，能否在推理时把一篇文档写进一小块权重，在原文离开之后，仍凭这块权重作答？

```text
文档 C ──写入──▶ ΔW_C ──问题 q（不带 C）──▶ 答案
```

## 可能性

如果成立，检索、重排、拼提示的整条管线会收成一步：读，然后写。提问不再背着原文。一份记忆就是一个文件，可以加载、卸下、叠加。智能体白天写下经历，夜里再把它们沉淀进更稳定的权重。

## 数学

**MLP 即键值表**（Geva 等，2021）。`z` 是键，`W_down` 的列是值。加上 `ΔW`，就是添条目。

```math
y = W_{\text{down}}\, z = \sum_j z_j\, w_j
```

**外积即联想记忆**（Kohonen 1972）。串扰随 `√(N/d_ff)` 增长。真实激活彼此相关，键需要一个度量把它们拉开：ROME 用 `C⁻¹`，ACWC 去学它。

```math
\Delta W = \sum_i a_i\, b_i^{\top},\quad b_i = \frac{k_i}{\lVert k_i\rVert^2}
\;\;\Rightarrow\;\;
\Delta W k_j = a_j + \sum_{i\neq j} a_i\,\frac{k_i^{\top}k_j}{\lVert k_i\rVert^2}
```

**梯度即外积**（Schmidhuber 1992；Sun 等，2024；Behrouz 等，2025）。推理时训练与联想记忆是同一件事。

```math
\frac{\partial \mathcal{L}}{\partial W} = \delta\, z^{\top}
```

**KV cache 与权重同源**（Schlag 等，2021）。一个逐 token 存，一个压成定长。

```math
\sum_i v_i\,(k_i^{\top} q) = \Big(\sum_i v_i k_i^{\top}\Big)\, q = S\, q
```

**两套学习系统**（McClelland 等，1995）。冻结的骨干是新皮层，`ΔW` 是海马体里的一段情景。

## 现状

受控合成文档（每篇四句“关系—值”，值为单 token）上，ACWC 的 rank-4 权重在原文移除后答对 79/96 道新组合问题、66/96 道训练时未见值的问题。同秩同范数的随机权重为 0/192。换上别的文档的权重，答出的是那篇文档的值。

自由文本、多 token 答案、多文档叠加，尚未触及。

## 设计

- **状态是权重。** 骨干冻结。每篇文档一份 `down_proj` 上的 `ΔW`，用完清零，以 BF16 落盘，可跨进程加载。
- **写读隔离。** 写入器只见原始字符串。题目在 `ΔW` 落盘后才出现。读出时只有问题。
- **等字节对比。** 8 层 `W_down` 共 8 × 2560 × 9728 个参数，每 token 的 KV 是 36 × 2 × 8 × 128，折合 2,702 个 KV token（`src/rlm/budget.py`）。
- **对照。** 不加载、加载、原文入提示、同秩同范数随机、错误文档，共五条臂。另有一组无关常识题，检查骨干是否受损。问答原样导出为 `.txt`。

## 算法

### 推理时梯度写入

只训练选中层的 `down_proj`，每 512 token 一步。目标可选 NTP、TTCD 或 DCD：冻结的教师看前文，带 `ΔW` 的学生不看。

```math
\mathcal{L}_{\text{DCD}} = \mathrm{KL}\big(p_{t}\,\|\,p_{s}\big) + \lambda\cdot\frac{1}{L}\sum_{l}\frac{\lVert h^{s}_{l}-h^{t}_{l}\rVert_1}{\lVert h^{t}_{l}\rVert_1}
```

`src/rlm/writer.py` · `src/rlm/methods.py` · `scripts/write.py` · `scripts/ask.py`

### 解析残差写入

固定探针 `P`。教师读 `[C; P]`，学生只读 `P`。取残差最大的 `r` 个位置：`X`（`d_ff × r`）是学生的 `down_proj` 输入，`R`（`d_model × r`）是教师与学生的输出之差。

```math
\Delta W = R\,(X^{\top}X+\lambda I)^{-1}X^{\top}
```

在探针子空间内拟合超过 99%，问题却读不出答案。写入的键和读取的键不在一处。

`scripts/analytic_residual.py`

### ACWC：读写对齐编译器

跨文档学一个共享编译器，新文档只需一次前向。逐句取键源 `k_i`（关系 token 的 `down_proj` 输入均值）和值源 `e_i`（句尾 token 在 `lm_head` 中的行，单位化）。

```math
\tilde{k}_i = (k_i\odot \exp(s))(I + UV^{\top}),\qquad
b_i = \frac{\tilde{k}_i}{\lVert\tilde{k}_i\rVert^2},\qquad
a_i = g\,e_i,\qquad
\Delta W_C = \sum_{i=1}^{m} a_i b_i^{\top} = A_C B_C
```

`b_iᵀ k̃_i = 1`，键对上就输出 `a_i`。值通道沿用模型自己的词向量，所以能写入训练时未见的值。读出时，`down_proj` 的输出加上 `A_C (B_C z)`。

外层训练只用合成文档，学习 `s`、`U, V`（秩 64）和 `g`：

```math
\mathcal{L} = \mathrm{CE}(\text{答案首 token}) + \lambda_{\text{route}}\,\mathrm{CE}\big(\mathrm{softmax}_i(\tau\cos(z_q,\tilde{k}_i)),\, i^{*}\big)
```

配置：`Qwen3-4B`，第 32 层，`m = 4`，`τ = 20`，`λ_route = 0.2`。编译器共 1,254,913 个参数，每篇文档 98,304 字节。

`scripts/acwc.py` · `scripts/acwc_summary.py`

## 碎碎念

墨尔本又进入了阴雨连绵的春夏交替。不上课的时候有时候我偶尔会想，如果有一天，AI 真的能做完所有人的工作，到那时我们还能剩下什么。

我不知道。有些夜里，这个问题让人睡不着。

幸运的是，它取代不了我的生活。工具之所以为工具，是因为它始终是方法和途径，而不是目的和结果。

地球上还有很多角落，我尚未探索，也不像互联网，没有被生成的内容占满。我们仍然可以是创作者，或哪怕只是探索者。

也许我想错了。也许明天一切都会不一样。

还好我们还拥有生活的全部，体验的权力。

## 许可证

[MIT](LICENSE)
