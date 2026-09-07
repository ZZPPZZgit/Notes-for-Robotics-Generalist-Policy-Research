# piSeries
## π0: A Vision-Language-Action Flow Model for General Robot Control
### Highlight:
a **novel flow matching architecture** built on top of a pre-trained vision-language model (VLM)
to inherit Internet-scale semantic knowledge 
- a novel generalist robot policy architecture based on VLM pre-training and flow matching
- an empirical investigation of pre-training/post training recipes for such robot foundation models 
### Background:
Solution: adopting a large-scale pre-training approach 
- Robot Learing: availability of data, generalization, and robustness 
- VLM: not truly situated in a physical world 
- NLP&CV: general-purpose foundation models tend to outperform narrowly tailored and specialized
solutions 

Challenges:
- done at a very large scale 
- right model architectures that can effectively make use of diverse data sources 
- relied heavily on delicate strategies for curating pre-training and post-training data

### Structure 
![项目截图](.\Images\pi0structure.png)

#### The $π_0$ Model Key Algorithms
data distribution: $p(A_t|o_t)$ \
action chunk of future actions: $A_t = [a_t, a_{t+1}, ..., a_{t+H−1}]$ 
- frequency: $H = 50$ 

observation: $o_t = [I^1_t, ..., I^n_t, ℓ_t, q_t]$ 
- $I^i_t$: $i^{th}$ image
- $ℓ_t$: a sequence of language tokens
- $q_t$: a vector of joint angles 
  
conditional flow matching loss: $L^τ(θ) = E_{p(A_t|o_t),q(A^τ_t|A_t)}||v_θ(A^τ_t, o_t) − u(A^τ_t|A_t)||^2$ 
 
Gaussian (or optimal transport) probability path: $q(A^τ_t|A_t) = N (τA_t, (1 − τ)I)$
- random noise: $ϵ ∼ N (0, I)$
- noisy actions: $A^τ_t = τA_t + (1 − τ)ϵ$
- network outputs: $v_θ(A^τ_t, o_t)$ 
- denoising vector field: $u(A^τ_t|A_t) = A_t − ϵ$

generate actions by integrating the
learned vector field from $τ = 0$ to $τ = 1$, starting with random
noise $A^0_t ∼ N (0, I)$  

$$A^τ_{t+δ} = A^τ_t + δv_θ(A^τ_t, o_t)$$
- $δ$ is the integration step size

#### Training Recipe
##### Pre-training
To down weight over-represented combinations: each task-robot combination weighted by $n^{0.43}$ 

For robots with lower dimensional configuration and action spaces: **zero-pad** the
configuration and action vectors

### Experimental Evaluation
research questions:
- How well does π0 perform after pre-training on a variety
of tasks that are present in the pre-training data?
- How well does π0 follow language commands?
- How does π0 compare to methods that have been proposed specifically for addressing dexterous manipulation tasks?
- Can π0 be adapted to complex, multi-stage tasks?

### Future Work
do not yet provide a comprehensive understanding of how the pre-training datasets should be composed.  
- it remains unclear how to predict how much and what kind of data is needed to attain near-perfect performance  

it remains to be seen how much positive transfer there is in combining highly diverse data particularly from different tasks and different robots  
- it is left for future work to understand whether this universality extends to much more distinct domains
