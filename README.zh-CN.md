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

### 目标

写入只见文档 `C`。带 `ΔW` 的模型只拿到问题，答得和把文档放进提示的同一模型一样，写入就成功了：

```math
J(\Delta W; C) = \mathbb{E}_{q\sim Q(C)} \sum_t \mathrm{KL}\big(p_\theta(a_t \mid C, q, a_{<t}) \,\big\|\, p_{\theta+\Delta W}(a_t \mid q, a_{<t})\big)
```

`Q(C)` 是关于 `C` 的问题分布。期望取在问题上，不在文档自己的 token 上。写入器看不到评测题，就自己采样：冻结的模型读 `C` 后自己出题，每句话再给一道完形填空。教师是把 `C` 放进提示的同一个冻结模型（Snell 等，2022；Caccia 等，2025；Eyuboglu 等，2025）。

### 输出投影即线性联想记忆

`y = W z`，`ΔW` 在每个位置加上 `ΔW z`（Geva 等，2021；Kohonen 1972）。把键排成 `K`（`d_in × n`），每个键需要的输出变化排成 `V`（`d_out × n`）。记 `Σ = E[z zᵀ]` 为通用文本上键的二阶矩，则 `tr(ΔW Σ ΔWᵀ)` 是 `ΔW` 在通用文本上造成的期望平方变化。写入求解

```math
\Delta W^\star = \arg\min_{\Delta W}\ \lVert \Delta W K - V \rVert_F^2 + \lambda\,\mathrm{tr}\big(\Delta W\,\Sigma\,\Delta W^\top\big)
= V\,(G + \lambda I)^{-1} K^\top \Sigma^{-1},\qquad G = K^\top \Sigma^{-1} K
```

这就是 MEMIT 的更新（Meng 等，2023）。由它推出三件事。

1. **问题只能通过与写入键的白化内积读出。** `ΔW* z = V (G + λI)⁻¹ (Kᵀ Σ⁻¹ z)`。若对每个写入键都有 `kᵢᵀ Σ⁻¹ z = 0`，读出就是零，写入对自己的键拟合得再好也没用。取自固定通用探针的键这样错过问题的键。取自文档位置的键也一样，所以在原文上做下一词预测存进去的事实，问题抽不出来（Allen-Zhu 与 Li，2023；Berglund 等，2023）。
2. **写入的键互不串扰。** `λ → 0` 且键线性无关时，`ΔW* K = V` 精确成立。赫布写入（对所有键求 `aᵢ kᵢᵀ / ‖kᵢ‖²` 之和）返回 `aⱼ`，再加上对 `i ≠ j` 求和的 `aᵢ kᵢᵀkⱼ / ‖kᵢ‖²`。这部分串扰随键的相关性增长（Hu 等，2024）。
3. **λ 有量纲。** 典型键满足 `kᵀ Σ⁻¹ k ≈ d_in`。取 `λ = κ · d_in`，它保留目标的 `1/(1+κ)`。不用 `Σ⁻¹` 时，真实的键挤在少数方向上：有效维度 `(tr Σ)² / tr(Σ²)` 远小于 `d_in`。

### 最优秩

`J` 是二次的，Hessian 为 `M = K Kᵀ + λ Σ`，所以 `J(ΔW) − J(ΔW*) = ‖(ΔW − ΔW*) M^{1/2}‖²_F`。最优的秩 `r` 写入是这个度量下的截断 SVD（Eckart–Young）：

```math
\Delta W_r = \big[\Delta W^\star M^{1/2}\big]_r\, M^{-1/2} = A B,\qquad
A = P_r S_r,\quad
B = R_r^\top\,\mathrm{diag}\Big(\tfrac{1}{\sqrt{g(g+\lambda)}}\Big)\, E^\top K^\top \Sigma^{-1}
```

其中 `G = E diag(g) Eᵀ`，`V E diag(√(g/(g+λ))) = P S Rᵀ`。`Σ⁻¹K` 之后的每一步都只涉及 `n × n` 矩阵。

### 梯度与 KV cache

对 `W` 的梯度是外积，`∂L/∂W = δ zᵀ`，所以梯度写入是同一个最小二乘的迭代解法（Schmidhuber 1992；Sun 等，2024；Behrouz 等，2025；Wang 等，2025）。线性注意力存 `S = V Kᵀ`、读 `S q`，KV cache 和权重是同一种记忆的两种形态（Schlag 等，2021）。

**两套学习系统**（McClelland 等，1995）。冻结的骨干是新皮层，`ΔW` 是海马体里的一段情景。

## 现状

ACWC，受控合成文档（每篇四句“关系—值”，值为单 token）：rank-4 权重在原文移除后答对 79/96 道新组合问题、66/96 道训练时未见值的问题。同秩同范数的随机权重为 0/192。换上别的文档的权重，答出的是那篇文档的值。这组数字来自按评测问法选出的检查点；代码现在按训练问法选。训练和测试用的是同样四种关系，所以结果说明的是值的绑定，不是为新关系写入键。

CCD 和基于自问的 DCD 通过了单元测试：闭式解、最优截断、写入键张成空间之外的键读出为零。它们在预训练模型上还没有结果。

## 设计

- **状态是权重。** 骨干冻结。每篇文档一份挂在 MLP 输出投影（`down_proj` 及其在各架构中的对应层）上的 `ΔW`，经 hook 生效，不改动原权重，用完清零，以 BF16 落盘，可跨进程加载。
- **写读隔离。** 写入器只见原始字符串，可以就这段字符串询问冻结的模型。评测题在 `ΔW` 落盘后才出现。读出时只有问题。
- **键统计量。** 每个记忆层的 `Σ` 取自模型自己采样的文本，或你提供的文件。特征分解缓存在 `outputs/stats/`。
- **等字节对比。** 8 层 `W_down` 共 8 × 2560 × 9728 个参数，每 token 的 KV 是 36 × 2 × 8 × 128，折合 2,702 个 KV token。任意模型可用 `ctw inspect` 换算。
- **对照。** 七条臂：不加载、原文入提示、加载、错误文档、同秩同范数随机、随机键配正确的值（保留 `A`）、正确的键配随机值（保留 `B`）。选择性 `log10 ‖ΔW z_q‖² / tr(ΔW Σ ΔWᵀ)` 衡量问题比通用文本多激活了多少记忆。另测与文档无关文本的困惑度，检查骨干是否受损。问答可原样导出为 `.txt`。

## 算法

### CCD：闭式上下文蒸馏

1. **自问。** 冻结的模型就 `C` 出题，每个事实问几种问法，每句话再加一道完形填空。教师的贪心答案接在两条序列后面，于是学生和教师共享问题、模板尾部和答案。
2. **目标。** 在最高一层记忆层上，给每个共享位置一个偏移 `δ`。它把学生的答案分布拉向教师，并贴近教师自己的残差 `r = h_teacher − h_student`：

```math
\delta^\star_q = \arg\min_\delta\ \frac{1}{|A_q|}\sum_{t\in A_q}\mathrm{KL}\big(p^{\text{teacher}}_t \,\|\, p^{\text{student}}_t(\delta)\big)
+ \beta\,\frac{1}{|S_q|}\sum_{t\in S_q}\frac{\lVert \delta_t - r_t\rVert^2}{\lVert h_t\rVert^2}
```

3. **求解。** 从最低的记忆层往上，`m` 层中的第 `i` 层承担最高层仍然缺少部分的等份，用上面的最优秩 `r` 写入拟合：

```math
V^{(i)} = \frac{h_0 + \delta^\star - h^{(i)}}{m - i + 1}
```

`h⁽ⁱ⁾` 是 `i` 以下各层写入后最高层的状态。问题之前的模板在每个提示里都一样，这些位置的目标为 0。

每次写入花费一次生成、每道题约 30 步对 `δ` 的梯度、每层每道题一次前向。不需要跨文档训练。答案可以是多个 token。

`src/ctw/writers/ccd.py` · `src/ctw/solve.py` · `src/ctw/queries.py` · `src/ctw/stats.py` · `configs/qwen3-4b-ccd.yaml`

### 基于自问的 DCD

同一目标的梯度解法（Caccia 等，2025）。每个记忆层上的低秩 `ΔW` 在自问题上训练：答案 token 上的 KL，加上在共享位置上、`ΔW` 能影响的每一层学生与教师输出的相对 L1。

```math
\mathcal{L}_{\text{DCD}} = \mathrm{KL}\big(p_{t}\,\|\,p_{s}\big) + \lambda\cdot\frac{1}{L}\sum_{l}\frac{\lVert h^{s}_{l}-h^{t}_{l}\rVert_1}{\lVert h^{t}_{l}\rVert_1}
```

`src/ctw/writers/gradient.py` · `configs/qwen3-0.6b-dcd.yaml`

### 原文上的 NTP

在文档上做下一词预测，每块一步。它的键是文档位置，按第 1 条，问题只能读到经相关键漏过来的部分。它是基线。

### ACWC：摊销编译器

跨文档学一个共享编译器，新文档只需一次前向。逐句取键源 `k_i`（关系 token 的 `down_proj` 输入均值）和值源 `e_i`（答案 token 在 `lm_head` 中的行，单位化）。

```math
\tilde{k}_i = (k_i\odot \exp(s))(I + UV^{\top}),\qquad
\Delta W_C = \sum_{i=1}^{m} g\,e_i\,\frac{\tilde{k}_i^{\top}}{\lVert\tilde{k}_i\rVert^2},\qquad
\Delta W_C\, z = \sum_{i=1}^{m} g\,e_i\,\frac{\tilde{k}_i^{\top} z}{\lVert\tilde{k}_i\rVert^2}
```

只有 `z = k̃ᵢ` 时才精确取回，而读出时从不产生这个 `z`。学到的度量和增益让问题的键 `z` 与正确的槽重叠。读出是 `m` 个槽上不带 softmax 的线性注意力（Liu 等，2026）。值通道沿用模型自己的词向量，所以能写入训练时未见的值。`ΔW` 在每个位置都加上同一个 `eᵢ`，所以只能写单 token 的值。

外层训练只用合成文档，学习 `s`、`U, V`（秩 64）和 `g`：

```math
\mathcal{L} = \mathrm{CE}(\text{答案首 token}) + \lambda_{\text{route}}\,\mathrm{CE}\big(\mathrm{softmax}_i(\tau\cos(z_q,\tilde{k}_i)),\, i^{*}\big)
```

配置：`Qwen3-4B`，第 32 层，`m = 4`，`τ = 20`，`λ_route = 0.2`。编译器共 1,254,913 个参数，每篇文档 98,304 字节。

`src/ctw/writers/acwc.py` · `configs/qwen3-4b-acwc.yaml`

## 用法

```bash
pip install -e .            # Python 3.10+；GPU、Apple MPS、CPU 均可

ctw inspect --model Qwen/Qwen3-0.6B --layers 0.89 --rank 4   # 层数、投影路径、状态字节、等价 KV token
ctw run configs/qwen3-0.6b-ccd.yaml                           # 自问 → 闭式写入 → 七臂评测
ctw run configs/qwen3-4b-ccd.yaml                             # CCD，用 ACWC 的同一批文档
ctw run configs/qwen3-4b-acwc.yaml --set seed=11              # ACWC；报告的数字用 7、11、19 三个种子
ctw summarize outputs/qwen3-4b-acwc/seed*.json --out outputs/qwen3-4b-acwc/summary.json
```

某个模型第一次跑 CCD 时，会采样文本计算键统计量并缓存。也可以用 `stats.source=text` 和 `stats.text=[...]` 指向你自己的文件。

单篇文档：

```bash
ctw write configs/qwen3-0.6b-ccd.yaml --document doc.txt --out doc.safetensors
ctw ask configs/qwen3-0.6b-ccd.yaml --state doc.safetensors --question "What was the name of the keeper's cat?"
```

ACWC 写入需要拟合好的编译器：加上 `--set writer.load=outputs/qwen3-4b-acwc/compiler-seed7.pt`。

**换模型、换设备。** 所有配置项都能用 `--set` 覆盖。

| 配置 | 作用 |
|---|---|
| `model.id` | Hugging Face 名称或本地路径 |
| `model.device` / `model.dtype` | `auto`、`cuda`、`mps`、`cpu` / `auto`、`bfloat16`、`float16`、`float32` |
| `model.layers_path` / `model.out_proj` | 自动识别失败时手动指定，如 `model.layers`、`mlp.down_proj` |
| `memory.layers` | `32`、`-4`、`0.89`（深度比例）、`last:8`、`[8, 16, 24]` |
| `writer.params.*` | 写入算法的超参，`ctw list` 列出默认值 |
| `stats.*` | 键统计量：`source`（`sample` 或 `text`）、`tokens`、`shrink`、`cache` |
| `eval.selectivity` | 每条臂报告 `log10 ‖ΔW z_q‖² / tr(ΔW Σ ΔWᵀ)` |
| `task.params.seen_values` | 换分词器后，键值任务的值须为单 token |

已测试：Qwen3、Llama、GPT-2、GPT-NeoX、OPT 的层结构。

**加一个算法。** 在 `src/ctw/writers/` 新建文件：

```python
from ctw.writers import Writer, register

@register("mine")
class MyWriter(Writer):
    def write(self, ctx, document):   # 只拿到原文
        ...                           # 返回 MemoryState
```

放在别处的文件用 `--set imports=[my_writer.py]` 加载。`--set writer.name=mine` 即可在同一套任务、对照和汇总下比较。跨文档学习的算法再实现 `fit()`。`ctx.stats.get(layer)` 给出某层的键统计量，`ctw.solve.covariance_ridge` 给出闭式写入。新任务用 `@register_task`；自己的文档和题目可直接写成 JSON，格式见 `examples/lighthouse.json`。

`pytest` 在随机初始化的小模型上跑完全部组件，不需要下载权重。

## 碎碎念

墨尔本又进入了阴雨连绵的春夏交替。不上课的时候有时候我偶尔会想，如果有一天，AI 真的能做完所有人的工作，到那时我们还能剩下什么。

我不知道。有些夜里，这个问题让人睡不着。

幸运的是，它取代不了我的生活。工具之所以为工具，是因为它始终是方法和途径，而不是目的和结果。

地球上还有很多角落，我尚未探索，也不像互联网，没有被生成的内容占满。我们仍然可以是创作者，或哪怕只是探索者。

也许我想错了。也许明天一切都会不一样。

还好我们还拥有生活的全部，体验的权力。

## 许可证

[MIT](LICENSE)
