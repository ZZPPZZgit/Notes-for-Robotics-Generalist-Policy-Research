# World Action Model

## DreamZero

### Highlights:
- learn physical
dynamics 
  - by predicting future world states and actions
  - using video as a dense representation of how the world evolves
- World Action Model: a foundation model designed to predict both actions and visual future states in an aligned manner

### Core Findings:
- Video quality determines policy performance
- Data diversity > volume
- Autoregressive advantage: 
  - autoregressive architectures yield smoother robot motions and higher modality alignment between predicted videos and executed actions

### Background
Video generation in robotics: vedio -> action

Joint video and action generation:
- adding a world modeling objective next to action prediction improves multi-task performance, sample efficiency, and generalization to novel scenes and objects
- earlier models learn joint world modeling + action prediction from scratch or from VLAs; recent ones build on pretrained video diffusion models to inherit rich visual dynamics priors
- World Action Model (WAM): a model that uses world modeling (predicting the future state) for action prediction
  - called WAM, not VAM: video is only one possible world modeling objective — future WAMs may align actions with tactile sensing, force feedback, or latent representations
- DreamZero: explores data diversity and scale, uses an autoregressive architecture for long-horizon world-action modeling

**Why WAMs:**
- built on video diffusion backbones, they inherit rich spatiotemporal priors from web-scale data — combining the seamless gradient flow of end-to-end VLAs with dense world modeling supervision for planning
- unlike latent world models, which learn dynamics from scratch, WAMs reuse pretrained video representations that already encode physical dynamics
- they learn the joint distribution of video and action: video prediction works as an implicit visual planner that guides action generation — so improving robot skills reduces to improving video generation
- three abilities beyond current VLAs: zero-shot generalization to novel tasks, learning from heterogeneous robot data, and efficient cross-embodiment transfer from videos

### Structure
![项目截图](Images/DreamZerostructure.png?raw=true)

#### Challenge and Solution
Video-action alignment: 
- train a single end-to-end
model that jointly denoises video and action with a shared objective

Architectural design: bidirectional($\pi$ action expert) or autoregressive
architectures are better suited for WAMs, with implications in modality alignment, error accumulation, and
inference efficiency; 
- autoregressive architecture
- replace predicted frames with ground-truth observations in the KV cache

Real-time inference: video diffusion models require iterative **denoising across
high-dimensional latent spaces**, making them prohibitively slow for closed-loop contro
- optimizations that achieve a 38× inference speedup

#### Architecture
$$\underbrace{\pi_\theta\big(\mathbf{o}_{l:l+H},\, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l},\, c,\,
  q_l\big)}_{\text{DreamZero}} \;=\; \underbrace{\pi_\theta\big(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l},\, c,\,
  q_l\big)}_{\text{video prediction}} \cdot \underbrace{\pi_\theta\big(\mathbf{a}_{l:l+H} \mid
  \mathbf{o}_{0:l+H},\, q_l\big)}_{\text{IDM}}$$
- train a single model end-to-end with joint prediction objective

introduce minimal additional parameters: 
- state encoders
- action encoders
- decoders
  
predict video frames and corresponding actions **autoregressively**
- predict video frames in a chunk manner; each chunk
has a fixed number of latent frames 𝐾 to match the action horizon
- bidirectional diffusion typically requires processing fixed-length sequences
  - cause mismatch between modality
- autoregressive modeling only for the video modality to avoid error propagation coming from closed-loop action prediction

Training Objective: flow-matching
- denoise the noisy current chunk conditioned on the clean previous chunks
- chunk index 𝑘 > 0 and the denoising timestep $𝑡_𝑘 ∈ [0, 1]$

noisy interpolations:
$$z^k_{t_k} = t_k\, z^k_1 + (1 - t_k)\, z^k_0, \qquad a^k_{t_k} = t_k\, a^k_1 + (1 - t_k)\, a^k_0$$
- $z^k_0, a^k_0 \sim \mathcal{N}(0, I)$: Gaussian noises; $z^k_1$: clean video latent; $a^k_1$: normalized action
- all frames within the same chunk share timestep $t_k$; different chunks get independent timesteps

clean context from the previous chunks:
$$\mathcal{C}_k = \{(z^j_1, a^j_1)\}_{j=1}^{k-1}$$

flow-matching objective:
$$\mathcal{L}(\theta) = \mathbb{E}_{z, a, \{t_k\}}\left[\frac{1}{K}\sum_{k=1}^{K} w(t_k)\,\Big\| u_\theta\big([z^k_{t_k}, a^k_{t_k}];\, \mathcal{C}_k, c, q^k, t_k\big) - v^k \Big\|^2\right]$$
- $u_\theta$: predicts the joint velocity for both modalities (video + action)
- $w(t_k) > 0$: predefined weight function; $c$: text condition; $q^k$: proprioceptive states of the $k$-th chunk
- target velocity: $v^k := [z^k_1, a^k_1] - [z^k_0, a^k_0]$
- video and action share the same denoising timestep within a chunk (faster convergence at the start of training)
- attention masking: the noisy current chunk attends to the clean context $\mathcal{C}_k$ of previous chunks

#### Real-time Execution
Challenge:**latency**
- iterative denoising across 16 diffusion steps required for smooth actions
- the computational cost of a 14B parameter DiT backbone
- sequential execution that blocks robot motion during inference

Solution:
- Asynchronous Closed-Loop Execution
- System-level Optimizations
  - CFG(Context-Free Grammar) Parallelism
  - DiT(Diffusion Transformer) Caching
    - if cosine similarity between successive velocities exceeds a threshold, reuse cached velocities
- Implementation-level Optimizations
  - Torch Compile and CUDA Graphs 
  - Post-Training Quantization
  - Kernel and Scheduler Enhancements.
- Model-level Optimizations: DreamZero-Flash
  - decoupling video and action noise schedules during training
  - orginal: a train-test mismatch during training
  - Training: use $𝑡^{video}_k = 1 − 𝜂$, $𝜂 ∼ Beta(𝛼, 𝛽)$biasing video timesteps toward high-noise states
  - exposes the
model to configurations where it must predict clean actions from noisy visual context
  = reduce the diffusion steps from four to one

### Training
- prioritizing task diversity
and real-world utility over task-specific repetition
- update all DiT blocks, the state encoder, action encoder,
and action decoder, while freezing the text encoder, image encoder, and VAE

### Experimental Evaluation
- Do WAMs learn better from diverse, non-repetitive data?
  - Most DreamZero failures stem from video generation errors rather than action prediction
- Do WAMs generalize to unseen tasks?
- Do WAMs improve post-training performance?
- Do WAMs enable strong cross-embodiment transfer to unseen tasks?
  - potential scaling pathway: abundant human video data—orders of magnitude larger than robot datasets—could enable WAMs to acquire diverse skills without action annotation
- Do WAMs enable few-shot new embodiment adaptation?
- Does DreamZero-Flash maintain performance with fewer denoising steps?
  - decoupled noise scheduling offers a more effective speed–accuracy trade-off for real-time deployment

### Future Work:
Scaling Laws of WAMs
Learning from In-the-wild Human Data
Faster Inference
Long-horizon Reasoning
High-Precision Tasks
Embodiment Design for WAMs