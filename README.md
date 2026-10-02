# Context-to-Weight Memory

[English](README.md) · [中文](README.zh-CN.md)

## Origin

Today's context engineering is bloated and convoluted, and I could not stand it. The human brain remembers without an external library. It relies on a distinct ability to internalize. If models show something like emergence past a certain scale, they must hold this potential for internalization too. That is where this project began.

Can a frozen model write a document into a small block of weights at inference time, and still answer from those weights after the text is gone?

```text
document C ──write──▶ ΔW_C ──question q (no C)──▶ answer
```

## What it could mean

If it holds, the whole pipeline of retrieval, reranking, and prompt assembly folds into one step: read, then write. Questions stop carrying the document. A memory becomes a file to load, unload, and stack. An agent writes down its day and, at night, settles it into more stable weights.

## Math

**An MLP is a key–value table** (Geva et al., 2021). `z` is the key, and the columns of `W_down` are the values. Adding `ΔW` adds entries.

```math
y = W_{\text{down}}\, z = \sum_j z_j\, w_j
```

**Outer products are associative memory** (Kohonen 1972). Crosstalk grows as `√(N/d_ff)`. Real activations are correlated, so keys need a metric that pulls them apart: ROME uses `C⁻¹`, and ACWC learns it.

```math
\Delta W = \sum_i a_i\, b_i^{\top},\quad b_i = \frac{k_i}{\lVert k_i\rVert^2}
\;\;\Rightarrow\;\;
\Delta W k_j = a_j + \sum_{i\neq j} a_i\,\frac{k_i^{\top}k_j}{\lVert k_i\rVert^2}
```

**A gradient is an outer product** (Schmidhuber 1992; Sun et al., 2024; Behrouz et al., 2025). Training at inference time and associative memory are the same thing.

```math
\frac{\partial \mathcal{L}}{\partial W} = \delta\, z^{\top}
```

**The KV cache and weights share one origin** (Schlag et al., 2021). One stores every token. The other compresses them to a fixed size.

```math
\sum_i v_i\,(k_i^{\top} q) = \Big(\sum_i v_i k_i^{\top}\Big)\, q = S\, q
```

**Two learning systems** (McClelland et al., 1995). The frozen backbone is the neocortex. `ΔW` is one episode in the hippocampus.

## Status

On controlled synthetic documents (four relation–value sentences each, single-token values), ACWC's rank-4 weights answer 79/96 questions on new combinations and 66/96 on values unseen in training, with the document removed. Random weights of the same rank and norm score 0/192. Load another document's weights, and the model answers with that document's values.

Free text, multi-token answers, and stacked documents are still untouched.

## Design

- **The state is weights.** The backbone is frozen. Each document gets one `ΔW` on `down_proj`, zeroed after use, saved as BF16, and loadable across processes.
- **Write and read stay apart.** The writer sees only the raw string. Questions appear after `ΔW` is saved. At read time, the prompt holds only the question.
- **Equal bytes.** 8 `W_down` layers hold 8 × 2560 × 9728 parameters. One token of KV holds 36 × 2 × 8 × 128. That equals 2,702 KV tokens (`src/rlm/budget.py`).
- **Controls.** Five arms: no write, write, document in prompt, random weights of the same rank and norm, and wrong-document weights. A set of unrelated general questions checks for backbone damage. Every exchange is exported as `.txt`.

## Algorithms

### Inference-time gradient writing

Only the selected `down_proj` layers train, one step per 512 tokens. The objective is NTP, TTCD, or DCD: a frozen teacher sees the earlier text, and the student with `ΔW` does not.

```math
\mathcal{L}_{\text{DCD}} = \mathrm{KL}\big(p_{t}\,\|\,p_{s}\big) + \lambda\cdot\frac{1}{L}\sum_{l}\frac{\lVert h^{s}_{l}-h^{t}_{l}\rVert_1}{\lVert h^{t}_{l}\rVert_1}
```

`src/rlm/writer.py` · `src/rlm/methods.py` · `scripts/write.py` · `scripts/ask.py`

### Analytic residual writing

Fixed probes `P`. The teacher reads `[C; P]`, and the student reads `P` alone. Take the `r` positions with the largest residual: `X` (`d_ff × r`) holds the student's `down_proj` inputs, and `R` (`d_model × r`) holds the teacher-minus-student outputs.

```math
\Delta W = R\,(X^{\top}X+\lambda I)^{-1}X^{\top}
```

It fits above 99% inside the probe subspace, yet questions cannot read it. The written keys and the reading keys live in different places.

`scripts/analytic_residual.py`

### ACWC: read–write aligned compiler

One compiler is learned across documents. A new document needs a single forward pass. For each sentence, take a key source `k_i` (mean `down_proj` input over the relation tokens) and a value source `e_i` (the unit-normalized `lm_head` row of the final token).

```math
\tilde{k}_i = (k_i\odot \exp(s))(I + UV^{\top}),\qquad
b_i = \frac{\tilde{k}_i}{\lVert\tilde{k}_i\rVert^2},\qquad
a_i = g\,e_i,\qquad
\Delta W_C = \sum_{i=1}^{m} a_i b_i^{\top} = A_C B_C
```

`b_iᵀ k̃_i = 1`, so a matching key outputs `a_i`. The value channel reuses the model's own embeddings, so it can write values never seen in training. At read time, the `down_proj` output gains `A_C (B_C z)`.

Outer training uses synthetic documents only and learns `s`, `U, V` (rank 64), and `g`:

```math
\mathcal{L} = \mathrm{CE}(\text{first answer token}) + \lambda_{\text{route}}\,\mathrm{CE}\big(\mathrm{softmax}_i(\tau\cos(z_q,\tilde{k}_i)),\, i^{*}\big)
```

Setting: `Qwen3-4B`, layer 32, `m = 4`, `τ = 20`, `λ_route = 0.2`. The compiler has 1,254,913 parameters, and each document takes 98,304 bytes.

`scripts/acwc.py` · `scripts/acwc_summary.py`

## Ramblings

Melbourne has slipped again into the long rains between spring and summer. On days without class, I sometimes wonder: if AI one day really can do everyone's work, what will we have left by then?

I do not know. Some nights, the question keeps me awake.

Luckily, it cannot replace my life. A tool is a tool because it stays a means and a path, never the end or the result.

There are still many corners of the earth I have not explored, and unlike the internet, they are not filled with generated content. We can still be creators, or even just explorers.

Maybe I am wrong. Maybe tomorrow everything changes.

Good thing we still hold the whole of life: the power to experience it.

## License

[MIT](LICENSE)
