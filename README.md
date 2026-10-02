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

### The objective

A write sees only the document `C`. It succeeds when the model with `ΔW`, given a question alone, answers as the same model does with the document in the prompt:

```math
J(\Delta W; C) = \mathbb{E}_{q\sim Q(C)} \sum_t \mathrm{KL}\big(p_\theta(a_t \mid C, q, a_{<t}) \,\big\|\, p_{\theta+\Delta W}(a_t \mid q, a_{<t})\big)
```

`Q(C)` is the distribution of questions about `C`. The expectation runs over questions, not over the document's own tokens. The writer cannot see the evaluation questions, so it samples its own: the frozen model reads `C` and writes questions about it, and each sentence also yields a cloze query. The teacher is the same frozen model with `C` in the prompt (Snell et al., 2022; Caccia et al., 2025; Eyuboglu et al., 2025).

### An out-projection is a linear associative memory

`y = W z`, and `ΔW` adds `ΔW z` at every position (Geva et al., 2021; Kohonen 1972). Collect keys `K` (`d_in × n`) and the output change each key must produce, `V` (`d_out × n`). Let `Σ = E[z zᵀ]` be the key second moment on generic text. Then `tr(ΔW Σ ΔWᵀ)` is the expected squared change `ΔW` makes on generic text, and the write solves

```math
\Delta W^\star = \arg\min_{\Delta W}\ \lVert \Delta W K - V \rVert_F^2 + \lambda\,\mathrm{tr}\big(\Delta W\,\Sigma\,\Delta W^\top\big)
= V\,(G + \lambda I)^{-1} K^\top \Sigma^{-1},\qquad G = K^\top \Sigma^{-1} K
```

This is MEMIT's update (Meng et al., 2023). Three facts follow from it.

1. **A question reads only through its whitened overlap with the written keys.** `ΔW* z = V (G + λI)⁻¹ (Kᵀ Σ⁻¹ z)`. If `kᵢᵀ Σ⁻¹ z = 0` for every written key, the readout is zero, however well the write fits its own keys. Keys taken from fixed generic probes miss question keys this way. So do keys taken from document positions, which is why next-token prediction on the text stores facts that questions cannot extract (Allen-Zhu & Li, 2023; Berglund et al., 2023).
2. **Written keys do not interfere.** As `λ → 0` with independent keys, `ΔW* K = V` exactly. The Hebbian write, the sum of `aᵢ kᵢᵀ / ‖kᵢ‖²` over keys, returns `aⱼ` plus the sum of `aᵢ kᵢᵀkⱼ / ‖kᵢ‖²` over `i ≠ j`. That crosstalk grows with key correlation (Hu et al., 2024).
3. **λ has a scale.** A generic key has `E[zᵀ Σ⁻¹ z]` equal to the sum of `λⱼ / λ̃ⱼ`, where `λⱼ` are the eigenvalues of `Σ` and `λ̃ⱼ = (1−s) λⱼ + s · mean(λ)` the shrunk ones the solver inverts. That is `d_in` without shrinkage and far less with it. With `λ = κ · E[zᵀ Σ⁻¹ z]`, a generic key keeps `1/(1+κ)` of its target. Without `Σ⁻¹`, real keys crowd into a few directions: the effective dimension `(tr Σ)² / tr(Σ²)` is far below `d_in`.

### The best rank

`J` is quadratic with Hessian `M = K Kᵀ + λ Σ`, so `J(ΔW) − J(ΔW*) = ‖(ΔW − ΔW*) M^{1/2}‖²_F`. The best rank-`r` write is the truncated SVD in that metric (Eckart–Young):

```math
\Delta W_r = \big[\Delta W^\star M^{1/2}\big]_r\, M^{-1/2} = A B,\qquad
A = P_r S_r,\quad
B = R_r^\top\,\mathrm{diag}\Big(\tfrac{1}{\sqrt{g(g+\lambda)}}\Big)\, E^\top K^\top \Sigma^{-1}
```

Here `G = E diag(g) Eᵀ` and `V E diag(√(g/(g+λ))) = P S Rᵀ`. Every step after `Σ⁻¹K` works on `n × n` matrices.

### Gradients and the KV cache

A gradient on `W` is an outer product, `∂L/∂W = δ zᵀ`, so a gradient write is an iterative solver of the same least squares (Schmidhuber 1992; Sun et al., 2024; Behrouz et al., 2025; Wang et al., 2025). Linear attention stores `S = V Kᵀ` and reads `S q`, which makes the KV cache and a weight two forms of one memory (Schlag et al., 2021).

**Two learning systems** (McClelland et al., 1995). The frozen backbone is the neocortex. `ΔW` is one episode in the hippocampus.

## Status

ACWC, on controlled synthetic documents (four relation–value sentences each, single-token values): its rank-4 weights answer 79/96 questions on new combinations and 66/96 on values unseen in training, with the document removed. Random weights of the same rank and norm score 0/192. Load another document's weights, and the model answers with that document's values. These numbers come from checkpoints selected on the evaluation phrasing. The code selects on the training phrasings. The four relations are the same in training and test, so the result shows value binding, not keys written for new relations.

CCD and DCD over self-queries pass the unit tests: the closed form, the optimal truncation, and the zero readout for keys outside the written span. They have no result on a pretrained model yet.

## Design

- **The state is weights.** The backbone is frozen. Each document gets one `ΔW` on the MLP output projection (`down_proj` or its counterpart in each architecture), applied through a hook without touching the original weights, zeroed after use, saved as BF16, and loadable across processes.
- **Write and read stay apart.** The writer sees only the raw string. It may ask the frozen model about that string. Evaluation questions appear after `ΔW` is saved. At read time, the prompt holds only the question.
- **Key statistics.** `Σ` for each memory layer comes from text the model samples, or from files you supply. Its eigendecomposition is cached under `outputs/stats/`.
- **Equal bytes.** 8 `W_down` layers hold 8 × 2560 × 9728 parameters. One token of KV holds 36 × 2 × 8 × 128. That equals 2,702 KV tokens. `ctw inspect` computes this for any model.
- **Controls.** Seven arms: no write, document in prompt, write, wrong-document weights, random weights of the same rank and norm, random keys behind the right values (`A` kept), and random values behind the right keys (`B` kept). Selectivity, `log10 ‖ΔW z_q‖² / tr(ΔW Σ ΔWᵀ)`, measures how much more a question activates the memory than generic text does. Perplexity on unrelated text checks for backbone damage. Every exchange can be exported as `.txt`.

## Algorithms

### CCD: closed-form context distillation

1. **Self-queries.** The frozen model writes questions about `C`, each fact asked several ways, plus one cloze query per sentence. The teacher's greedy answer is appended to both sequences, so student and teacher share the question, the template tail, and the answer.
2. **Targets.** At the top memory layer, an offset `δ` on every shared position. It moves the student's answer distribution to the teacher's and stays near the teacher's own residual `r = h_teacher − h_student`:

```math
\delta^\star_q = \arg\min_\delta\ \frac{1}{|A_q|}\sum_{t\in A_q}\mathrm{KL}\big(p^{\text{teacher}}_t \,\|\, p^{\text{student}}_t(\delta)\big)
+ \beta\,\frac{1}{|S_q|}\sum_{t\in S_q}\frac{\lVert \delta_t - r_t\rVert^2}{\lVert h_t\rVert^2}
```

3. **Solve.** From the lowest memory layer up, layer `i` of `m` takes an equal share of what is still missing at the top layer, and the best rank-`r` write above fits it:

```math
V^{(i)} = \frac{h_0 + \delta^\star - h^{(i)}}{m - i + 1}
```

`h⁽ⁱ⁾` is the top layer's state with the layers below `i` already written. The template before the question is the same in every prompt, so its positions get the target 0.

Each write costs one generation, about 30 gradient steps on `δ` per query, and one forward per query per layer. No training across documents. Answers can span several tokens.

`src/ctw/writers/ccd.py` · `src/ctw/solve.py` · `src/ctw/queries.py` · `src/ctw/stats.py` · `configs/qwen3-4b-ccd.yaml`

### DCD over self-queries

The gradient solver of the same objective (Caccia et al., 2025). Low-rank `ΔW` on every memory layer trains on the self-queries: KL on the answer tokens plus relative L1 between student and teacher outputs of every layer `ΔW` can change, over the shared positions.

```math
\mathcal{L}_{\text{DCD}} = \mathrm{KL}\big(p_{t}\,\|\,p_{s}\big) + \lambda\cdot\frac{1}{L}\sum_{l}\frac{\lVert h^{s}_{l}-h^{t}_{l}\rVert_1}{\lVert h^{t}_{l}\rVert_1}
```

`src/ctw/writers/gradient.py` · `configs/qwen3-0.6b-dcd.yaml`

### NTP on the document text

Next-token prediction on the document, one step per chunk. Its keys are document positions, so by fact 1 a question reads only what leaks through correlated keys. It is the baseline.

### ACWC: amortized compiler

One compiler is learned across documents. A new document needs a single forward pass. For each sentence, take a key source `k_i` (mean `down_proj` input over the relation tokens) and a value source `e_i` (the unit-normalized `lm_head` row of the answer token).

```math
\tilde{k}_i = (k_i\odot \exp(s))(I + UV^{\top}),\qquad
\Delta W_C = \sum_{i=1}^{m} g\,e_i\,\frac{\tilde{k}_i^{\top}}{\lVert\tilde{k}_i\rVert^2},\qquad
\Delta W_C\, z = \sum_{i=1}^{m} g\,e_i\,\frac{\tilde{k}_i^{\top} z}{\lVert\tilde{k}_i\rVert^2}
```

Recall is exact only at `z = k̃ᵢ`, which the read never produces. The learned metric and gain make the question's key `z` overlap the right slot. The readout is linear attention over `m` slots without a softmax (Liu et al., 2026). The value channel reuses the model's own embeddings, so it can write values never seen in training. `ΔW` adds the same `eᵢ` at every position, so it writes single-token values only.

Outer training uses synthetic documents only and learns `s`, `U, V` (rank 64), and `g`:

```math
\mathcal{L} = \mathrm{CE}(\text{first answer token}) + \lambda_{\text{route}}\,\mathrm{CE}\big(\mathrm{softmax}_i(\tau\cos(z_q,\tilde{k}_i)),\, i^{*}\big)
```

Setting: `Qwen3-4B`, layer 32, `m = 4`, `τ = 20`, `λ_route = 0.2`. The compiler has 1,254,913 parameters, and each document takes 98,304 bytes.

`src/ctw/writers/acwc.py` · `configs/qwen3-4b-acwc.yaml`

## Usage

```bash
pip install -e .            # Python 3.10+; GPU, Apple MPS, or CPU

ctw inspect --model Qwen/Qwen3-0.6B --layers 0.89 --rank 4   # layers, projection path, state bytes, KV-token equivalent
ctw run configs/qwen3-0.6b-ccd.yaml                           # self-queries → closed-form write → seven-arm evaluation
ctw run configs/qwen3-4b-ccd.yaml                             # CCD on the documents ACWC is measured on
ctw run configs/qwen3-4b-acwc.yaml --set seed=11              # ACWC; its reported numbers use seeds 7, 11, 19
ctw summarize outputs/qwen3-4b-acwc/seed*.json --out outputs/qwen3-4b-acwc/summary.json
```

The first CCD run on a model samples text for the key statistics and caches them. Point `stats.source=text` and `stats.text=[...]` at your own files instead.

One document:

```bash
ctw write configs/qwen3-0.6b-ccd.yaml --document doc.txt --out doc.safetensors
ctw ask configs/qwen3-0.6b-ccd.yaml --state doc.safetensors --question "What was the name of the keeper's cat?"
```

ACWC writes need its fitted compiler: add `--set writer.load=outputs/qwen3-4b-acwc/compiler-seed7.pt`.

**Other models, other devices.** Every config value can be overridden with `--set`.

| Key | Meaning |
|---|---|
| `model.id` | Hugging Face name or local path |
| `model.device` / `model.dtype` | `auto`, `cuda`, `mps`, `cpu` / `auto`, `bfloat16`, `float16`, `float32` |
| `model.layers_path` / `model.out_proj` | set by hand when detection fails, e.g. `model.layers`, `mlp.down_proj` |
| `memory.layers` | `32`, `-4`, `0.89` (depth fraction), `last:8`, `[8, 16, 24]` |
| `writer.params.*` | writer hyperparameters; `ctw list` shows defaults |
| `stats.*` | key statistics: `source` (`sample` or `text`), `tokens`, `shrink`, `cache` |
| `eval.selectivity` | report `log10 ‖ΔW z_q‖² / tr(ΔW Σ ΔWᵀ)` per arm |
| `task.params.seen_values` | with another tokenizer, key–value values must be single tokens |

Tested layouts: Qwen3, Llama, GPT-2, GPT-NeoX, OPT.

**Add an algorithm.** Create a file in `src/ctw/writers/`:

```python
from ctw.writers import Writer, register

@register("mine")
class MyWriter(Writer):
    def write(self, ctx, document):   # receives only the raw text
        ...                           # returns a MemoryState
```

A file kept elsewhere loads with `--set imports=[my_writer.py]`. `--set writer.name=mine` compares it under the same tasks, controls, and summaries. Writers that learn across documents also implement `fit()`. `ctx.stats.get(layer)` gives a layer's key statistics, and `ctw.solve.covariance_ridge` gives the closed-form write. New tasks use `@register_task`; your own documents and questions can go straight into JSON, as in `examples/lighthouse.json`.

`pytest` runs every component on tiny randomly initialized models, with no weights to download.

## Ramblings

Melbourne has slipped again into the long rains between spring and summer. On days without class, I sometimes wonder: if AI one day really can do everyone's work, what will we have left by then?

I do not know. Some nights, the question keeps me awake.

Luckily, it cannot replace my life. A tool is a tool because it stays a means and a path, never the end or the result.

There are still many corners of the earth I have not explored, and unlike the internet, they are not filled with generated content. We can still be creators, or even just explorers.

Maybe I am wrong. Maybe tomorrow everything changes.

Good thing we still hold the whole of life: the power to experience it.

## License

[MIT](LICENSE)
