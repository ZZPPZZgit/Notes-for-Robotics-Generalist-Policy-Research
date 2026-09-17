# CVAE (Conditional Variational Autoencoder)

> 起因:ACT 的核心生成模型就是 CVAE。本篇从 AE → VAE → CVAE 一步步推,最后对照 ACT 的 Algorithm 1,弄清每个设计"为什么"。
> 关键词:variational inference / ELBO / reparameterization trick / multimodality / style variable

---

## 1. 动机:为什么模仿学习需要生成模型

**问题:演示数据是多模态的 (multimodal)**
- 同一任务,人类演示有多种"做法":左手先动 or 右手先动、从左边绕 or 从右边绕、快 or 慢
- 即给定同一观测 $o$,演示动作的分布 $p(a \mid o)$ 有多个"峰"(modes)

**普通 behavioral cloning 的失败方式**
- BC 通常用 MSE/L1 回归 $\hat{a} = f_\theta(o)$ —— 输出的是分布的**条件均值**
- 双峰各在左边和右边时,均值落在中间 → 动作既不左也不右 → 失败(mode averaging)
- 神经网络能拟合多模态分布,但前提是损失函数允许:必须**显式建模分布**而非只输出单个点

**解法方向(都是"输出分布"而非"输出点")**
- CVAE:引入隐变量 $z$,一次采样决定走哪个 mode —— ACT 的选择
- Diffusion:迭代去噪拟合多模态分布 —— Diffusion Policy / pi0 系列的选择
- Autoregressive/discretization:把动作离散化逐维预测 —— 不需要额外隐变量

---

## 2. 从 AE 到 VAE

### 2.1 Autoencoder 的局限
- encoder $z = f(x)$,decoder $\hat{x} = g(z)$,训练目标重构损失
- 问题 1:$z$ 是**确定性**的,只能重构、不能**生成** —— 随机采一个 $z$ 丢给 decoder,输出大概率是噪声
- 问题 2:latent space 没有结构,训练时见过的 $z$ 区域之外是"荒地"

### 2.2 VAE:把生成过程写成概率模型
VAE 定义一个显式的生成模型(先验 + 似然):

$$z \sim \mathcal{N}(0, I), \qquad x \sim p_\theta(x \mid z)$$

目标:最大化数据的对数似然。对所有 $z$ 积分(edge likelihood):

$$\log p_\theta(x) = \log \int p_\theta(x \mid z)\, p(z)\, dz$$

- 这个积分**不可解**(对每个 $z$ 都要过一遍 decoder);后验 $p_\theta(z \mid x)$ 同样不可解

### 2.3 变分推断:用一个网络近似后验
引入 $q_\phi(z \mid x)$(encoder)去近似真实后验 $p_\theta(z \mid x)$。可以推出恒等式:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big] - D_{KL}\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{ELBO}(x)} + \; D_{KL}\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big)$$

- 右边第二项(KL 到真后验)不可算但**非负** → ELBO 是 $\log p_\theta(x)$ 的**下界**(这就是名字的由来:Evidence Lower BOund)
- 最大化 ELBO 一石二鸟:既提高数据似然,又让 $q_\phi$ 逼近真后验

**ELBO 的两项直觉解读**
- 重构项 $\mathbb{E}_q[\log p_\theta(x \mid z)]$:采出的 $z$ 要能还原 $x$ → $z$ 携带 $x$ 的信息
- KL 项 $D_{KL}(q_\phi(z\mid x) \,\|\, \mathcal{N}(0,I))$:每个样本的 $z$ 分布别离先验太远 → latent space 整体被"压"到 $\mathcal{N}(0, I)$ 附近
- 两项天然对抗:前者想多存信息,后者想少存信息 → $z$ 是一个**信息瓶颈**

### 2.4 Reparameterization trick
- loss 需要对 $z \sim q_\phi$ 采样,但"采样"这个操作不可导,梯度传不到 $\phi$
- 技巧:把随机性挪到外部

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

- 现在 $z$ 对 $\mu, \sigma$ 是确定性的可导函数,梯度畅通

### 2.5 术语卡:prior / likelihood / posterior

**先验 (prior) $p(z)$** —— 看到任何观测**之前**,对 $z$ 分布的假设("先" = 逻辑上先于观测,非时间上先)。VAE 里它是**人为选定**的 $p(z) = \mathcal{N}(0, I)$,不是从数据估出来的。

| 名称 | 记号 | 含义 |
|---|---|---|
| 先验 prior | $p(z)$ | 看到 $x$ **之前**,$z$ 的分布("出厂信念") |
| 似然 likelihood | $p(x \mid z)$ | $z$ 给定时 $x$ 的分布(生成方向) |
| 后验 posterior | $p(z \mid x)$ | 看到 $x$ **之后**反推的 $z$ 分布("修正后的信念") |

$$\underbrace{p(z \mid x)}_{\text{后验}} = \frac{\overbrace{p(x \mid z)}^{\text{似然}}\ \overbrace{p(z)}^{\text{先验}}}{p(x)} \qquad \text{即 后验} \propto \text{似然} \times \text{先验}$$

- 直觉:先验 = 出厂信念,观测到的证据通过似然修正它 → 后验;数据越多,先验影响越弱
- 非 ML 例子:来路不明的硬币,先验 $\theta \sim \text{Beta}(5,5)$("大概公平")→ 见到 10 投 9 正 → 后验向 $\theta \approx 0.8$ 偏移

**为什么 VAE 选 $\mathcal{N}(0, I)$ —— 三个后果,正好对应后文三处设计**
1. 推断时可采样:运行时没有演示、算不了后验,但随时能从先验采 $z$ 喂 decoder(§4.5)
2. KL 有闭式解:先验与后验同为(对角)高斯,才有 §6 末尾的闭式公式
3. KL 项的意义:训练用后验 $q_\phi$ 采样、推断用先验采样,KL 把两者拉近 → 两个世界合一(§3.3)

**CVAE 的条件先验 $p(z \mid c)$**:看到 $c$ 但未见 $x$ 时 $z$ 的分布;ACT 取无条件的简化版 $p(z) = \mathcal{N}(0, I)$ → Algorithm 1 第 10 行 KL 里写的是 $\mathcal{N}(0, I)$,不依赖 $\bar{o}_t$

---

## 3. CVAE:给 VAE 加条件

### 3.1 改动:所有分布都多一个条件 $c$
CVAE (Sohn et al. 2015) 建模的是**条件分布** $p_\theta(x \mid c)$:

| 组件 | VAE | CVAE |
|---|---|---|
| 先验 | $p(z) = \mathcal{N}(0, I)$ | $p(z \mid c)$(常仍取 $\mathcal{N}(0, I)$) |
| 似然/decoder | $p_\theta(x \mid z)$ | $p_\theta(x \mid z, c)$ |
| 近似后验/encoder | $q_\phi(z \mid x)$ | $q_\phi(z \mid x, c)$ |

$$\text{ELBO}(x, c) = \mathbb{E}_{q_\phi(z \mid x, c)}\big[\log p_\theta(x \mid z, c)\big] - D_{KL}\big(q_\phi(z \mid x, c) \,\|\, p(z \mid c)\big)$$

### 3.2 $z$ 在 CVAE 里学到什么(核心直觉)
- decoder 同时看 $c$ 和 $z$ → **$c$ 能解释的规律性,全部记在 $c$ 头上**
- $z$ 只需要编码 $c$ 解释不了的**残差变化** —— 正是"同一个 $o$ 下多种做法"的多模态信息
- 对模仿学习:$c = o_t$(任务/场景),$z = $ 完成方式(轨迹风格、模式选择)

### 3.3 训练 vs 推断:两套角色
```
训练(encoder + decoder,用 ELBO):
  x, c ──► q_φ(z|x,c) ──► μ,σ ──► z(重参数化)──┐
                                                ▼
  c ─────────────────────────────► p_θ(x|z,c) ──► x̂
  loss = 重构项 + β·KL(q_φ(z|x,c) ‖ N(0,I))

推断(只用 decoder):
  z ~ N(0, I)(不需要 x!)──► p_θ(x|z,c) ──► 生成的 x
```
- encoder 是**训练脚手架**:它看到 $x$(要生成的目标),负责告诉 decoder"这次是哪种 mode",让 decoder 学会条件于 $z$ 的完整分布
- KL 项的意义就在这里:把 $q_\phi$ 拉向 $\mathcal{N}(0,I)$,保证推断时从先验采的 $z$ 落在 decoder 训练时见过的分布内 —— **没有 KL 项,训练和推断就是两个世界**

---

## 4. CVAE 在 ACT 中的实例化

### 4.1 变量对应
- 生成的 $x$ = 动作块 $a_{t:t+k}$($k$ 步动作组成的序列)
- 条件 $c$ = 当前观测 $o_t$(4 相机图像 + 双臂关节位置)
- $z$ = style variable,官方实现默认 $\in \mathbb{R}^{32}$

$$q_\phi\big(z \mid a_{t:t+k},\, \bar{o}_t\big) \qquad \pi_\theta\big(\hat{a}_{t:t+k} \mid o_t,\, z\big)$$

### 4.2 一个刻意的非对称:encoder 不看图像
- $\bar{o}_t$ = $o_t$ **去掉图像**(只剩关节位置)
- 动机:若 encoder 能看到图像,$z$ 会倾向编码场景内容;限制成"动作 + 本体感知"后,$z$ 被逼着只编码**风格**(怎么做),而不是**场景**(在哪做)
- 场景信息 decoder 自己能从 $o_t$ 拿到,不需要 $z$ 转发

### 4.3 Transformer 实现
- **encoder** $q_\phi$:输入序列 = $[\texttt{CLS}] + \bar{o}_t + a_{t:t+k}$ 的 token 化;取 $\texttt{CLS}$ 输出过线性头 → 高斯的 $\mu, \sigma$
- **decoder** $\pi_\theta$:观测 token 化(图像过 CNN 出 patch token、关节位置过线性层)+ $z$ 作为 query 前缀;transformer decoder **并行**输出 $k$ 步动作(DETR 式,非自回归)

### 4.4 损失(对照 Algorithm 1 第 9-11 行)
$$L = L_{\text{reconst}} + \beta\, L_{\text{reg}}$$

- $L_{\text{reconst}} = \big\|\hat{a}_{t:t+k} - a_{t:t+k}\big\|_2^2$ —— 即 ELBO 重构项的简化:固定方差的高斯似然下,负对数似然 ∝ MSE,常数项丢弃
- $L_{\text{reg}} = D_{KL}\big(q_\phi(z \mid a_{t:t+k}, \bar{o}_t) \,\|\, \mathcal{N}(0, I)\big)$ —— ELBO 的 KL 项
- $\beta$:超参。论文消融 $\beta \in \{0.1, 1, 10, 100\}$,**$\beta = 10$ 最好**(更强地压 latent,防 $z$ 编码过多信息导致推断 mismatch)
- $\beta > 1$ 相当于 $\beta$-VAE 式的加权 KL:加大信息瓶颈强度

### 4.5 推断(机器人运行时)
1. 只保留 decoder $\pi_\theta$(encoder 整个弃用)
2. 每步:取 $z = 0$ → $\hat{a}_{t:t+k} = \pi_\theta(\hat{a}_{t:t+k} \mid o_t, z)$(论文 Algorithm 2 第 4 行明确写 $z=0$;官方代码 `get_action` 同样置零)
   - 标准 CVAE 做法是从先验采样 $z \sim \mathcal{N}(0, I)$;ACT 直接取先验的众数 $z = 0$ —— 确定性、可复现,且恰在 decoder 训练时所见 $z$ 分布的中心($\beta = 10$ 的强 KL 已把后验压得很近先验)
3. $z$ 的作用主要在**训练阶段**(style 监督信号、吸收 $o_t$ 解释不了的多模态残差);推断时置零 → 策略输出确定
4. (chunk 重叠执行与 temporal ensemble 是另一层机制,不属于 CVAE;完整流程见 [ActionChunk.md](ActionChunk.md) 的 Algorithm 2)

---

## 5. 常见疑问 FAQ

**Q1:训练时 encoder 能"看到答案"($a_{t:t+k}$),这不是作弊吗?**
不是。encoder 近似的是后验 $p(z \mid a, o)$ —— "已知这帧动作,它属于哪种风格"。它只参与训练,作用是给 decoder 提供监督信号,让 decoder 学会**条件于 $z$ 的完整多模态分布**。推断时没有答案,自然也没有 encoder。

**Q2:为什么推断时从 $\mathcal{N}(0,I)$ 采样就有效?**
因为 KL 项在训练中把每个样本的后验 $q_\phi$ 都拉向 $\mathcal{N}(0,I)$ → decoder 训练时见到的 $z$ 基本覆盖先验采样区域。KL 项就是训练分布与推断分布之间的桥。
(实际中 ACT 更简单:直接取 $z = 0$ —— 先验的众数,确定性且位于分布中心,见 §4.5。)

**Q3:KL 项会不会太强,把 $z$ 压成没信息?**
会,叫 **posterior collapse**:$\beta$ 过大 → $q_\phi \to \mathcal{N}(0,I)$,$z$ 与 $x$ 独立 → decoder 忽略 $z$ → 退化成普通确定性回归,多模态能力丢失。$\beta$ 是"信息瓶颈松紧"旋钮:ACT 在其任务上 $\beta=10$ 是甜点。

**Q4:CVAE vs Diffusion Policy?**
目标相同(建模多模态动作分布),手段不同:CVAE 用**一次隐变量采样** + 单步解码;diffusion 用**迭代去噪**(几十步)逼近分布。diffusion 表达能力更强、训练更稳;CVAE 推断快(一次前向)。这也是 ACT → Diffusion Policy → pi0 的路线演进逻辑之一。

**Q5:$z$ 里到底存了什么,可视化过吗?**
论文称之为 style variable:同一任务不同演示的轨迹形状差异(速度、接近方向、抓取策略等)。因为 encoder 被剥夺了图像输入,它能依赖的只有"这段动作本身的写法"。

---

## 6. 最小伪代码(PyTorch 风格)

```python
# ---------- 训练一步(对应 Algorithm 1 第 7-11 行) ----------
mu, logvar = encoder(action_chunk, obs_proprio)   # q_φ(z | a, ō):CLSpooled transformer
z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterization
action_hat = decoder(z, obs_full)                 # π_θ(â | o, z):图像+关节 token,z 作 query 前缀

L_recon = F.mse_loss(action_hat, action_chunk)
L_reg   = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()  # KL(q‖N(0,I)) 闭式解
loss    = L_recon + beta * L_reg                  # ACT: beta = 10

# ---------- 推断(机器人在线运行) ----------
z = torch.randn(batch, z_dim)                     # z ~ N(0, I);encoder 整个弃用
action_chunk = decoder(z, obs_full)
```

KL 闭式解(对角高斯 vs 标准正态):

$$D_{KL} = \tfrac{1}{2} \sum_{j} \big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\big)$$

---

## 7. 参考文献
- Sohn et al., *Learning Structured Output Representation using Deep Conditional Generative Models*, NeurIPS 2015 —— CVAE 原始论文
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014 —— VAE / reparameterization
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023 —— ACT,§IV 的 CVAE 实例化(即本仓库 `ACT.pdf`,笔记见 [ActionChunk.md](ActionChunk.md))
