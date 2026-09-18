# Diffusion Policy
## Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

### Highlights:
introduces Diffusion Policy
- a new way of generating robot behavior 
- representing a robot’s visuomotor policy as a conditional denoising diffusion process
- incorporation of receding horizon control,
visual conditioning, and the time-series diffusion transformer
- Closed-loop action sequences: receding-horizon control
  - predict an **action chunck**, execute and **execute only** the first action
- Visual conditioning
  - visual observations treated as conditioning instead of a part of the joint data distribution
- Time-series diffusion transformer

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

  - $$A^t_{k-1} = \alpha\big(A^t_k - \gamma\,\epsilon_\theta(O^t, A^t_k, k) +
  \mathcal{N}(0, \sigma^2 I)\big)$$
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
