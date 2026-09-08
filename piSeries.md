# piSeries
## $π_0$: A Vision-Language-Action Flow Model for General Robot Control
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
![项目截图](Images/pi0structure.png?raw=true)

#### Architecture
data distribution: $p(A_t|o_t)$ \
action chunk of future actions: $A_t = [a_t, a_{t+1}, ..., a_{t+H−1}]$ 
- horizon: $H = 50$ 

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

Incorporating the flow matching timestep: For each noisy action $a^τ_{t′}$ ,corresponding embedding that is fed into the
transformer is $W_3 · swish(W_2 · concat(W_1 · a^τ_{t′}, ϕ(τ)))$
- $ϕ:  \mathbb{R} \mapsto \mathbb{R}^w$ sinusoidal positional encoding function
- $W_1 ∈ \mathbb{R}^{w×d}$ ; $W_2 ∈ \mathbb{R}^{w×2w}$ ; $W_3 ∈ \mathbb{R}^{w×w}$
- $concat(W_1 ⋅ a^τ_{t′},ϕ(τ)) ∈ \mathbb{R}^{2w}$ connect 
- $swish()$ active function
- $w$ embedding dimension; $d$ action dimension

Attention mask: π0 uses a blockwise causal attention mask
with 3 blocks: $[I, ℓ_t], [q_t], and [a^τ_t, ..., a^τ_{t+H−1}]$
- the tokens in each block **cannot** attend to the tokens in future blocks

Action expert: 
- each token is routed to one of the experts
- the weights interact only through the transformer’s self-attention layers
Sampling the flow matching timestep:
- a timestep sampling distribution that emphasizes low timesteps (high noise levels)

Inference:
- model takes an observation ot and the noisy actions $A^τ_t$
- outputs the vector field that needs to be integrated to obtain the next
flow matching step, $v_t^τ$
- encode each of the images $I$, run a forward pass on the tokens corresponding to $o_t$
- then run 10 steps of flow matching, each step requires running a
forward pass on the tokens corresponding to $A^τ_t$
#### Training Recipe
##### Pre-training
To down weight over-represented combinations: each task-robot combination weighted by $n^{0.43}$ 

For robots with lower dimensional configuration and action spaces: **zero-pad** the configuration and action vectors

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


## $π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization

### Highlight:
A system for training a highly
generalizable VLA
- The system uses a combination of **co-training** and **hybrid multi-modal examples** that combine image observations, language commands, object detections, semantic subtask prediction, and low-level actions

### Background:
Generalist robot manipulation policies:
- broadening the training data distribution: allows the resulting policies to not only solve a wider range of tasks out of the
box, but also improves their ability to generalize to new
scenes and tasks
- VLAs are still
typically evaluated in environments that closely match their
training data

Non-robot data co-training:
- using diverse non-robot data to improve the generalization of robot policies

Robot reasoning and planning with language:
- augmenting end-to-end policies
with high-level reasoning can significantly improve performance for long-horizon tasks
- two separate models for this purpose

Robotic learning systems with open-world generalization:
- methods do not readily generalize to the
full range of possible tasks that a generalist robot might need
to perform
- tasks in these demonstrations are still relatively simple

### Structure
![项目截图](Images/pi0.5structure.png?raw=true)

#### Training recipe
##### Pre-training
- intended to adapt the model to diverse robotic tasks
- data:
  - Diverse Mobile Manipulator data
  - Diverse Multi-Environment non-mobile robot data
  - Cross-Embodiment laboratory data
  - High-Level subtask prediction
  - Multi-modal Web Data
- trained as a
standard VLM transformer model by mapping actions to text
tokens (α = 0)
- trained as a standard auto-regressive
transformer, performing next-token prediction of text, object
locations, and FAST encoded action tokens.
##### Post-training
- intended to specialize it to mobile manipulation and equip it
with the mechanisms for efficient test-time inference
- adapt the model to also have an action expert
- add additional action expert weights
predicting continuous action tokens in a non-autoregressive
fashion for fast inference 
- optimize the objective in Equation (1), with
α = 10.0 for 80k additional steps
#### Architecture
distribution captured by the model: $π_θ(a_{t:t+H}, \hat{l} |o_t, ℓ) = π_θ(a_{t:t+H}|o_t, \hat{l})π_θ(\hat{ℓ}|o_t, ℓ)$

multimodal input tokens: $x_{1:N}$

output: $y_{1:N} = f(x_{1:N}, A(x_{1:N}), ρ(x_{1:N}))$
- $A(x_{1:N}) ∈ [0, 1]^{N×N}$ attention matrix indicating if a token can attend to another token
- $ρ(x_i)$ token type, desiding encoder 
- the output of f is split into text token logits and action output tokens, respectivel $y(y^l_{1:M}, y^a_{1:H})$

combined loss: $E_{D,τ,ω}[H(x_{1:M}, f_θ^ℓ(o_t, ℓ))+ α \lVert ω - a_{at:t+H} - f_θ^a(a^{τ,ω}_{t:t+H}, o_t, ℓ)\rVert^2]$
- $H(x_{1:M}, y^ℓ_{1:M})$: the cross entropy loss between the
text tokens and predicted logits (including the FAST encoded
action tokens)

a separate MLP for projecting τ only and then applies adaptive RMSNorm to inject the timestep information to each layer of the action expert

### Experimental Evaluation
Research Questions:
- Can π0.5 effectively generalize to complex multi-stage
tasks in entirely new homes?
- How does the generalization of π0.5 scale with the
number of distinct environments in the training data?
- How do the individual co-training ingredients in the π0.5
training mixture contribute to its final performance?
- How does π0.5 compare to the π0 VLA?
- How important is the high-level inference component of
π0.5, and how does it compare to flat, low-level inference
as well as oracle high-level baselines?

### Future Work
Some environments present persistent challenges, some behaviors present challenges with
partial observability, and in some cases the high-level subtask inference is easily distracted.

The model also uses a relatively modest
context, and incorporating richer context and memory could
make the model significantly more capable in settings with
more partial observability

specific sources of data can be explored even more broadly