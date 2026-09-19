# Diffusion Policy
## Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

### Highlights:
introduces Diffusion Policy
- a new way of generating robot behavior 
- representing a robot’s visuomotor policy as a conditional denoising diffusion process
- incorporation of receding horizon control
visual conditioning, and the time-series diffusion transformer
- Closed-loop action sequences: receding-horizon control
  - predict an **action chunck**, execute and **execute only** the first action
- Visual conditioning
  - visual observations treated as conditioning instead of a part of the joint data distribution
- Time-series diffusion transformer

### Background:
behavior cloning approaches categorized by policy structure:
Explicit Policy
- observation to action, direct regression
- one forward pass to simple & fast
- **not multimodal**, weak at high-precision tasks
- workarounds:
  - discretize action space 
  - Categorical + Gaussian mixtures
    - hyperparameter sensitive
    - mode collapse
    - limited precision

Implicit Policy (IBC)
- EBM assigns each action an **energy**
- find min-energy action
  - different actions can share low energy 
  -  **naturally multimodal**
- **unstable to train** 

Diffusion Models in control
- learning the **gradient field (score)** of an
  implicit energy, optimized at inference
- prior uses: 
  - planning: open-loop trajectory inference 
  - RL 
  - augmenting
- this work: diffusion **as the visuomotor BC policy**
  - new time-series transformer for action diffusion
  - integration of visual observations as conditioning

### Sturcture:
![项目截图](Images/DiffusionPolicyStructure.png?raw=true)
#### Architecture:
Denoising Diffusion Probabilistic Models
- $x^K$ sampled from Gaussian noise
  - Square Cosine Noise Schedule
- denoising $x^k, x^{k-1} \ldots x^0$
- $$x^{k-1} = \alpha\big(x^{k} - \gamma\,\epsilon_\theta(x^{k}, k) + \mathcal{N}(0, \sigma^2 I)\big) $$
- a single noisy gradient descent step: $x' = x - \gamma\nabla E(x)$
- loss function:$\mathcal{L} = MSE(ε^k, ε_θ (x^0 +ε^k, k))$

Closed-loop action-sequence prediction:
- $T_o$ as the observation horizon
- $T_p$ as the action prediction horizon
- $T_a$ as the action execution horizon.

Visual observation conditioning:
  - approximate $p(A^t|O^t)$ instead of joint $p(A^t,O^t)$ 
    - joint: diffuse $(O,A)$ as one trajectory → must **infer future states**, vision
  encoder runs at **every** denoising step
    - conditional: only $A^t$ is diffused, $O^t$ is a **fixed condition** (like class
  conditioning in image diffusion)
    - vision encoder runs **once** regardless of denoising iterations → faster, real-time
  inference

  - $$A^t_{k-1} = \alpha\big(A^t_k - \gamma\,\epsilon_\theta(O^t, A^t_k, k) + \mathcal{N}(0, \sigma^2 I)\big)$$
  - only change is $\epsilon_\theta(x^k, k) \to \epsilon_\theta(O^t, A^t_k, k)$
  — conditioning enters via **network input**, diffusion process unchanged

neural network architectures for $ε_θ$
- CNN：performs poorly when
 the desired action sequence **changes quickly and sharply**
 through time
- Time-series diffusion transformer:
  - $O_t$ is transformed into observation embedding sequence by a shared MLP
  - $\epsilon_{\theta}$ is predicted by each corresponding output token of the decoder stack
  - more sensitive to hyperparameters

#### Training Stability 
- IBC: in theory **same advantages** as diffusion (implicit, multimodal), but
  inherently **unstable to train** 
  
- implicit policy = EBM:
  $$p_\theta(a|o) = \frac{e^{-E_\theta(o,a)}}{Z(o,\theta)}, \quad Z(o,\theta)=\int e^{-E_\theta(o,a)}\,da$$
  - $Z(o,\theta)$: **intractable** normalization constant (integral over the whole
    continuous action space) — must be estimated to train
- InfoNCE loss estimates $Z$ with **negative samples**:
  $$\mathcal{L}_{InfoNCE} = -\log\frac{e^{-E_\theta(o,a)}}{e^{-E_\theta(o,a)} + \sum_{j=1}^{N_{neg}} e^{-E_\theta(o,\tilde{a}_j)}}$$
  - finite negatives cover only a tiny slice of action space → **biased estimate of $Z$**
  - energy net can "cheat": lower energy near positives while digging **spurious
    low-energy basins** where negatives don't cover
  - loss decreases smoothly but the distribution is wrong; sampling falls into bad
    regions → gradient spikes → instability (Du et al. 2020; Ta et al. 2022)
- Diffusion Policy sidesteps $Z$ by modeling the **score function**:
  $$\nabla_a \log p(a|o) = -\nabla_a E_\theta(a,o) - \underbrace{\nabla_a \log Z(o,\theta)}_{=0} \approx -\epsilon_\theta(a,o)$$
  - $Z(o,\theta)$ does **not depend on $a$** → $\nabla_a \log Z = 0$

- one-liner: instability comes from **estimating $Z$ with finite negative samples**,
  not from EBM expressiveness; diffusion learns the **same** distribution via the
  $Z$-free score → stable training- 

### Core Findings:
Diffusion Policy can express short-horizon multimodality
- achieving the same immediate goal

Diffusion Policy can express long-horizon multimodality
- completion of different sub-goals

Diffusion Policy can better leverage position control
- compared to speed control
  
The tradeoff in action horizon
- too long a horizon reduces performance due to slow reaction time

Robustness against latency

Diffusion Policy is stable to train

### Future Work:
inherits limitations from behavior cloning
- Diffusion policy can be applied to other paradigms, such as reinforcement learning

diffusion policy has higher computational costs and inference latency

#### Relationship with Future Work
##### piSeries
[piSeriesNotes](piSeries.md)

| Policy | Progress | Limitation&Improvment |
|---|---|---|
|DiffusionPolicy| incorporate DiT to generate action sequence| high inference cost |
|FlowMatching| modify the DiT: the noise are added in Gaussian probability path, $A^τ_t = τA_t + (1 − τ)ϵ$ | liner noise: lower inference cost |
