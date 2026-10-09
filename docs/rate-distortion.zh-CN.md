# 上下文写入权重的率失真分析

冻结的模型把上下文 $C$ 写成一块状态 $\Delta$，之后只凭 $\Delta$ 回答问题。本文回答三个问题：

1. 结构不变时，$\Delta$ 能放进已有层，还是必须新加一层？
2. 一块给定大小的 $\Delta$ 最少丢多少信息？
3. 压缩率和失真怎样取舍？

结论先列在这里，推导在后文。

- 失真有下界：$D \ge I(C;Y\mid Q) - R$。状态每多一比特，问答损失最多少一比特。
- 需要的比特数不超过模型给上下文的码长 $-\log_2 p_\theta(C)$。模型越强，码长越短，同样大小的 $\Delta$ 装得越多。这是"压缩即智能"在这个问题上的具体形式。
- 低秩 $\Delta$ 有三本账：信源要多少比特，容器能存多少比特，读出能取多少比特。秩决定读出带宽，参数量决定存储量。
- 线性化后，秩-失真曲线有闭式解。取舍规则是反向注水：保留能量高于阈值 $\theta$ 的奇异方向，第 $j$ 个方向分配 $\tfrac12\log_2(\sigma_j^2/\theta)$ 比特。
- 本仓库的 ACWC 结果用了约 $7.9\times10^5$ 比特的状态，任务只需 15.4 比特。瓶颈不在存储量，而在写入键与读出键的对齐。

## 1. 设定

| 记号 | 含义 |
|---|---|
| $\theta$ | 冻结的骨干参数 |
| $C$ | 上下文（本仓库任务里是一篇文档），服从分布 $\pi$ |
| $E$ | 写入器，$\Delta = E(C)$，只看 $C$ |
| $Q=(q_1,\dots,q_N)$ | 写入之后才出现的问题 |
| $Y=(y_1,\dots,y_N)$ | 答案，由教师 $p_\theta(y\mid C,q)$ 生成 |
| $p_{\theta,\Delta}(y\mid q)$ | 学生：带状态 $\Delta$、不带 $C$ 的模型 |
| $R$ | 率：存储 $\Delta$ 的比特数，$H(\Delta)\le R$ |
| $D$ | 失真：学生相对教师的超额对数损失 |

失真按比特计：

```math
D \;=\; \mathbb{E}\sum_{i=1}^{N}\mathrm{KL}\big(p_\theta(\cdot\mid C,q_i)\,\big\|\,p_{\theta,\Delta}(\cdot\mid q_i)\big).
```

另有一项干扰失真 $D_{\text{off}}$。它度量 $\Delta$ 对无关输入的改变，本仓库用无关文本的困惑度测它。

### 1.1 已有层与新层在数学上的关系

本仓库把 $\Delta W$ 挂在 MLP 输出投影上。设 $z$ 是 `down_proj` 的输入：

```math
y = (W_{\text{down}} + \Delta W)\,z = W_{\text{down}}\,z + A\,(B\,z),\qquad \Delta W = AB,\; A\in\mathbb{R}^{d_{\text{out}}\times r},\; B\in\mathbb{R}^{r\times d_{\text{in}}}.
```

这个式子有两种读法，函数完全相同：

- **改已有层。** 把 $\Delta W$ 并入 $W_{\text{down}}$。结构不变，计算量不变。
- **加一个并行的线性层。** 它读已有特征 $z$，输出加回残差。

所以"写进已有层"等于"加一个读已有特征的线性新层"。两者的真正差别只在三处：

1. 读出特征：线性层只能读 $z$；带非线性的新层读 $\phi(Bx)$，能分开 $z$ 中线性不可分的键。
2. 门控：并入的 $\Delta W$ 对所有输入都生效；带门控的新层可以对无关输入输出零。
3. 合并：线性 $\Delta W$ 能并入原权重；非线性新层不能，结构随之改变。

第 5 节给出这三处差别各自的代价。

## 2. 信源端：需要多少比特

**定理 1（逆定理）。** 对任何只看 $C$ 的写入器，有

```math
D \;\ge\; I(C;Y\mid Q) - R .
```

**证明。** 学生的预测分布只依赖 $(\Delta,Q)$，交叉熵不小于条件熵：

```math
\mathbb{E}\big[-\log_2 p_{\theta,\Delta}(Y\mid Q)\big] \;\ge\; H(Y\mid \Delta,Q) \;=\; H(Y\mid Q) - I(Y;\Delta\mid Q) \;\ge\; H(Y\mid Q) - R .
```

最后一步用 $I(Y;\Delta\mid Q)\le H(\Delta\mid Q)\le H(\Delta)\le R$。教师的对数损失恰为 $H(Y\mid C,Q)$。两式相减，得 $D\ge H(Y\mid Q)-H(Y\mid C,Q)-R = I(C;Y\mid Q)-R$。$\square$

定理 1 说明三件事：

- $R=0$（不写入）时，下界是 $I(C;Y\mid Q)$，即上下文给答案带来的全部信息。学生恰为教师对 $C$ 取平均时取等。
- 状态每多一比特，失真最多降一比特。
- 无损（$D=0$）至少需要 $R \ge I(C;Y\mid Q)$ 比特。

**与"压缩即智能"的联系。** 需要的比特数有上界：

```math
I(C;Y\mid Q) \;\le\; H(C) \;\le\; \mathbb{E}_{C\sim \pi}\big[-\log_2 p_\theta(C)\big].
```

右边是模型用算术编码压缩 $C$ 的平均码长。它比 $H(C)$ 多出 $\mathrm{KL}(\pi\,\|\,p_\theta)$，所以对任何上下文分布都成立。模型能预测的部分不必写入，只有意外的部分要占状态。模型越强，码长越短，同一块状态能装下的上下文越长。只有问题要求复现整段上下文时，两边才接近。实际问题只问一部分，所以需要的比特数通常远小于码长。

最优写入器的目标是信息瓶颈（Tishby 等，1999）：

```math
\min_{E}\; I(C;\Delta) \;-\; \beta\, I(\Delta;Y\mid Q).
```

它只保留与未来问题相关的信息。率失真曲线 $D(R)$ 就是这条瓶颈曲线。

## 3. 容器端：能存多少比特

秩 $r$ 的 $d_{\text{out}}\times d_{\text{in}}$ 矩阵构成一个流形，自由度为

```math
P(r) = r\,(d_{\text{in}} + d_{\text{out}} - r).
```

每个参数存 $b$ 比特，容器上限是 $R_{\text{store}} = b\,P(r)$。BF16 存储时名义上 $b=16$。Allen-Zhu 与 Li（2024）在训练得到的语言模型中测得约 2 比特/参数的知识容量。这个数来自他们的实验，本仓库没有测过。

存储端很宽松。以 Qwen3-4B 第 32 层、$r=4$ 为例，$P \approx 4\times(9728+2560) = 49{,}152$。按 2 比特/参数，它能装约 6,000 个 16 比特的事实。真正的限制在读出端。

## 4. 读出端：能取多少比特

状态写好以后，冻结的网络只通过问题位置的激活 $z_q$ 读它。读出要做两件事：找到对的槽位，再把槽位里的值译成 token。

### 4.1 寻址容量：秩是读出带宽

**模型假设。**

- 上下文有 $m$ 个槽位（事实），问题问的是槽位 $s$，$s$ 在 $m$ 个槽位上均匀分布。
- 问题位置的激活是 $z = \mu_s + n$，$n\sim\mathcal{N}(0,S_w)$ 与 $s$ 独立。$S_w$ 是同一槽位不同问法之间的协方差，$S_b=\mathrm{Cov}(\mu_s)$ 是槽位之间的协方差。
- 答案由记忆分支的输出 $A(Bz)$ 决定。冻结的下游只负责把值向量译成 token。ACWC 取 $a_i = g\,e_i$，正是这种设计。

**定理 2（寻址容量）。** 对任意 $B\in\mathbb{R}^{r\times d_{\text{in}}}$，

```math
I(s;Bz) \;\le\; \tfrac12\log_2\det\!\big(I + (BS_wB^{\top})^{-1}BS_bB^{\top}\big) \;\le\; C_r = \sum_{j=1}^{r}\tfrac12\log_2(1+\lambda_j),
```

其中 $S_w$ 正定，$\lambda_1\ge\lambda_2\ge\cdots$ 是广义特征值问题 $S_b u = \lambda S_w u$ 的特征值。

**证明。** $Bz$ 的协方差是 $B(S_b+S_w)B^\top$，同协方差下高斯分布的熵最大，所以 $h(Bz)\le \tfrac12\log_2\det\big(2\pi e\,B(S_b+S_w)B^\top\big)$。给定 $s$ 时 $Bz$ 是高斯分布，$h(Bz\mid s)=\tfrac12\log_2\det(2\pi e\,BS_wB^\top)$。两者相减得第一个不等式。行列式在 $r$ 维投影上的最大值由前 $r$ 个广义特征值取到，得第二个不等式。$\square$

由 Fano 不等式，槽位错误率为 $\varepsilon$ 时

```math
(1-\varepsilon)\log_2 m \;\le\; C_r + 1 .
```

这个结果和高斯信道容量 $\tfrac12\log_2(1+\mathrm{SNR})$ 同形。$B$ 的每一维是一条子信道，$\lambda_j$ 是它的信噪比。

由此得到三条推论：

1. **秩是读出带宽。** $B$ 决定能分开多少槽位，$A$ 决定每个槽位存什么值。秩只限制前者。
2. **最优的 $B$ 是 Fisher 判别方向。** 它取 $S_w^{-1}S_b$ 的前 $r$ 个特征向量，不是激活方差最大的方向。ROME 用 $C^{-1}k$ 做白化，ACWC 学习度量 $(k\odot e^{s})(I+UV^\top)$，路由损失直接优化判别性，三者都在提高 $\lambda_j$。
3. **写入键必须落在读出的判别子空间里。** 解析写入器用探针位置的激活作键。问题位置的 $\mu_s$ 在这些方向上没有差别，对应 $\lambda_j\approx 0$。所以它在探针子空间内拟合超过 99%，问题却读不出答案。

以 $r=4$、各方向 $\lambda_j = 3$ 为例，$C_r = 4$ 比特。即使 $\varepsilon\to 0$，也只能分开 $m\le 2^{5}=32$ 个槽位。同样的 $\lambda_j$ 下，$r=64$ 时上限是 $2^{65}$。

### 4.2 外积写入的串扰

外积写入取 $\Delta W=\sum_i a_i b_i^\top$，$b_i = k_i/\lVert k_i\rVert^2$。用 $k_j$ 读出：

```math
\Delta W k_j = a_j + \sum_{i\ne j} c_{ij}\,a_i,\qquad c_{ij} = \frac{k_i^\top k_j}{\lVert k_i\rVert^2}.
```

设键的协方差为 $\Sigma$。在有效维数 $d_{\text{eff}}$ 较大时，串扰总能量的一阶近似为

```math
\mathbb{E}\sum_{i\ne j} c_{ij}^2 \;\approx\; \frac{m-1}{d_{\text{eff}}},\qquad d_{\text{eff}} = \frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}.
```

真实激活的谱衰减很快，$d_{\text{eff}}$ 远小于 $d_{\text{in}}$。把键白化成 $\Sigma^{-1/2}k$ 后，$d_{\text{eff}}=d_{\text{in}}$，串扰降到 $(m-1)/d_{\text{in}}$。这就是 README 中"键需要一个度量把它们拉开"的定量形式。

## 5. 率失真曲线

### 5.1 线性化后的闭式解

在选定层上线性化。取 $N$ 个读出位置：

- $Z\in\mathbb{R}^{d_{\text{in}}\times N}$：学生在这些位置的 `down_proj` 输入。
- $R_{\text{res}}\in\mathbb{R}^{d_{\text{out}}\times N}$：教师输出减学生输出的残差。
- $\Sigma_g$：无关文本上 $z$ 的协方差，用来度量干扰。

目标是问答失真加干扰失真，约束是秩：

```math
\min_{\operatorname{rank}\Delta\le r}\; J(\Delta)=\lVert \Delta Z - R_{\text{res}}\rVert_F^2 + \beta\,\operatorname{tr}(\Delta\,\Sigma_g\,\Delta^\top).
```

第二项是并入的 $\Delta W$ 在无关输入上的平均输出能量 $\mathbb{E}_x\lVert\Delta W z_x\rVert^2$。

**定理 3（秩-失真）。** 令 $M = ZZ^\top + \beta\Sigma_g$（设其正定），$G = R_{\text{res}}Z^\top M^{-1/2}$，$\sigma_1\ge\sigma_2\ge\cdots$ 是 $G$ 的奇异值。则

```math
\Delta^{*}_r = [G]_r\,M^{-1/2},\qquad D(r) = J(\Delta^{*}_r) = \lVert R_{\text{res}}\rVert_F^2 - \sum_{j=1}^{r}\sigma_j^2 .
```

$[G]_r$ 是 $G$ 的前 $r$ 项截断奇异值分解。

**证明。** 展开得 $J(\Delta)=\operatorname{tr}(\Delta M\Delta^\top) - 2\operatorname{tr}(\Delta ZR_{\text{res}}^\top) + \lVert R_{\text{res}}\rVert_F^2$。代入 $\Gamma = \Delta M^{1/2}$，得 $J = \lVert \Gamma - G\rVert_F^2 + \lVert R_{\text{res}}\rVert_F^2 - \lVert G\rVert_F^2$。$M^{1/2}$ 可逆，所以 $\operatorname{rank}\Gamma=\operatorname{rank}\Delta$。由 Eckart–Young 定理，$\Gamma^* = [G]_r$。$\square$

$D(r)$ 分成两部分：

```math
D(r) \;=\; \underbrace{\lVert R_{\text{res}}\rVert_F^2 - \lVert G\rVert_F^2}_{D_\infty} \;+\; \underbrace{\sum_{j>r}\sigma_j^2}_{\text{截断损失}} .
```

- $D_\infty$ 是任何秩都去不掉的部分。它包含 $z$ 线性读不出的残差，以及为压低干扰付出的代价。$D_\infty$ 大时，加秩没有用，要换读出特征（第 5.3 节）。
- 截断损失由 $G$ 的谱尾决定。谱衰减越快，小秩越够用。
- $\beta=0$ 时，定理 3 退化为降秩回归（Izenman，1975）。本仓库的解析写入器是它的特例：$N=r$ 个探针位置，$\Sigma_g=I$，$\beta=\lambda$。

### 5.2 取舍规则：反向注水

秩和精度都花比特。第 $j$ 个奇异方向占 $n = d_{\text{in}}+d_{\text{out}}$ 个参数。设它每个参数存 $b_j$ 比特，在高分辨率量化近似下，量化误差约为 $\sigma_j^2\,2^{-2b_j}$。总的率和失真为

```math
R = n\sum_j b_j,\qquad D \approx D_\infty + \sum_j \sigma_j^2\,2^{-2b_j}.
```

对 $D+\mu R$ 求极小，得到反向注水：

```math
b_j = \max\!\Big(0,\;\tfrac12\log_2\frac{\sigma_j^2}{\theta}\Big),\qquad D(\theta) \approx D_\infty + \sum_j \min(\sigma_j^2,\theta).
```

阈值 $\theta$ 由比特预算定。规则读作：

1. 把奇异方向按 $\sigma_j^2$ 从大到小排列。
2. 保留 $\sigma_j^2 > \theta$ 的方向，这就是秩。
3. 给第 $j$ 个方向分配 $\tfrac12\log_2(\sigma_j^2/\theta)$ 比特/参数。
4. 丢掉其余方向，每个方向的损失是 $\sigma_j^2$。

在最优点，最后一比特换来的失真下降等于 $\mu$。在这个近似下，曲线 $D(R)$ 单调不增且为凸。$b_j=0$ 时误差恰为 $\sigma_j^2$，与丢掉该方向一致。这个形式和高斯向量信源的率失真函数一致。区别在于秩的基向量也要存，所以每个方向有固定开销 $n$。

### 5.3 已有层、新层与 KV cache 的取舍

| 容器 | 状态大小 | 串扰 | 对无关输入的干扰 | 结构 |
|---|---|---|---|---|
| 线性 $\Delta W$（已有层） | 固定，$r(d_{\text{in}}+d_{\text{out}})$ | $\approx (m-1)/d_{\text{eff}}$ | $\operatorname{tr}(\Delta W\Sigma_g\Delta W^\top)$，不能关 | 不变，可并入 |
| 带门控的新层 $A\,\phi(Bx-\tau)$ | 固定，与线性层同阶 | 门控切掉远处的键 | 门未打开时为零 | 改变 |
| KV cache / softmax 注意力 | 随 $m$ 线性增长 | 键分得开时随 $d$ 指数下降 | 为零 | 不变，但不压缩 |

线性记忆和 KV cache 是同一个读出式的两端：

```math
\text{线性：}\;\Big(\sum_i v_i k_i^\top\Big) q, \qquad \text{softmax：}\;\sum_i v_i\,\frac{e^{\,\tau k_i^\top q}}{\sum_l e^{\,\tau k_l^\top q}} .
```

线性形式把所有对压成一个定长矩阵，代价是串扰。softmax 形式没有串扰，代价是保存每一对 $(k_i,v_i)$，状态随 $m$ 增长。现代 Hopfield 网络（Ramsauer 等，2020）给出 softmax 形式的指数容量。

结构不变时，还有第三种做法：写入 MLP 的输入侧 $W_{\text{gate}}$、$W_{\text{up}}$。SwiGLU 的门控提供非线性寻址，层的形状不变。代价是它没有定理 3 那样的闭式解。

## 6. 本仓库的数

ACWC 在 Qwen3-4B 第 32 层写入秩 4 的状态，每篇上下文含 4 个槽位。

**状态大小。**

- 参数：$4\times(9728+2560)=49{,}152$。BF16 存储为 98,304 字节，即 786,432 比特。
- 一个 token 的 KV：$2\times36\times8\times128 = 73{,}728$ 个参数，即 147,456 字节。
- 一篇 $n$ 个 token 的上下文，相对 KV cache 的压缩比为 $147{,}456\,n / 98{,}304 = 1.5\,n$。

**需要的比特。** 4 个值从 16 个已见值中无放回抽取，$I(C;Y\mid Q)=\log_2(16\cdot15\cdot14\cdot13)=15.4$ 比特。未见值从 8 个值中抽取，为 $\log_2(8\cdot7\cdot6\cdot5)=10.7$ 比特。

**信息效率。** $\eta = 15.4/786{,}432 \approx 2.0\times10^{-5}$。即使按 2 比特/参数计，$\eta\approx1.6\times10^{-4}$。

**读出的比特。** 由 Fano 不等式，每题准确率 $1-\varepsilon$、候选值 $V$ 个时，每个答案至少取回 $\log_2 V - h(\varepsilon) - \varepsilon\log_2(V-1)$ 比特，$h$ 是二元熵。

| 测试集 | 准确率 | 每个答案的信息量 | 至少取回 |
|---|---|---|---|
| 已见值新组合 | 79/96 | 4.0 比特 | 2.63 比特 |
| 未见值 | 66/96 | 3.0 比特 | 1.23 比特 |

状态比任务所需大约五万倍，答案却只取回四成到三分之二的比特。所以瓶颈不在存储量。ACWC 每个槽位独占一个秩方向，秩只用来寻址，取回多少由 $\lambda_j$ 决定（第 4.1 节）。按这个分析，下一步应提高问题位置的 $\lambda_j$，而不是增加参数。这一点尚未经实验确认。

## 7. 测量 $D(R)$ 的步骤

评测已经记录每条臂的金标答案平均对数概率。`context` 臂是教师，`write` 臂是学生。

1. 对每个问题，取 $\hat D = (\log p_{\text{context}} - \log p_{\text{write}})/\ln 2$，单位是比特。答案有多个 token 时，用总对数概率，不用平均值。
2. 对每个写入器配置，记下 $R = 16\times$ 状态参数数。
3. 在同一张图上画出各配置的 $(R,\hat D)$，以及下界 $D = [I(C;Y\mid Q)-R]_+$。
4. 记下 $\hat D$ 与下界的距离。这段距离就是写入器和读出损失的比特数。
5. 在问题位置估计 $S_b$、$S_w$，算出 $\lambda_j$ 和 $C_r$，检查 Fano 界是否先于存储界收紧。
6. 用无关文本的困惑度变化估计 $D_{\text{off}}$，在 $(R,\hat D,D_{\text{off}})$ 三维上比较配置。

## 参考文献

- Shannon, C. E. (1959). Coding theorems for a discrete source with a fidelity criterion.
- Tishby, N., Pereira, F., Bialek, W. (1999). The information bottleneck method.
- Izenman, A. J. (1975). Reduced-rank regression for the multivariate linear model.
- Kohonen, T. (1972). Correlation matrix memories.
- Ramsauer, H. 等 (2020). Hopfield networks is all you need.
- Meng, K. 等 (2022). Locating and editing factual associations in GPT（ROME）.
- Geva, M. 等 (2021). Transformer feed-forward layers are key-value memories.
- Allen-Zhu, Z., Li, Y. (2024). Physics of language models: Part 3.3, knowledge capacity scaling laws.
- Delétang, G. 等 (2024). Language modeling is compression.
