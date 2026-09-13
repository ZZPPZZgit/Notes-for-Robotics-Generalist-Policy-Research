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

## $\pi^*_{0.6}$: a VLA That Learns From Experience
### Highlight:
A VLA model that
can improve through real-world deployments via reinforcement
learning (RL)
- a general-purpose method, RL with
Experience and Corrections via Advantage-conditioned Policies
(RECAP)

### Supervised Learning vs. Reinforcement Learning (generated by Deepseek V4.0 Fast)

**The one-line difference:**
- In **Supervised Learning**, the model is given a static dataset of *correct* input–output pairs and learns to imitate them (it is told the answer).
- In **Reinforcement Learning**, the agent is given *no correct answers* — it interacts with an environment, receives a scalar reward for the outcome of its behavior, and improves through trial and error.

**Side-by-side comparison:**

| Dimension | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Feedback signal | Instructive labels (ground-truth answer per sample) | Evaluative scalar reward (only "good/bad", never "what to do") |
| Data source | Pre-collected static dataset (often human demonstrations) | Sequentially collected by the agent's *own* interaction with the environment |
| Data structure | Samples assumed i.i.d. and fixed | Sequential (state → action → reward → next state, an MDP); non-stationary — the distribution shifts as the policy changes |
| Decision structure | Single-step mapping $x \to y$ | Sequential decision making with long-horizon consequences |
| Objective | Minimize prediction loss (e.g., cross-entropy, MSE) | Maximize cumulative discounted return $E[\sum_t \gamma^t r_t]$ |
| Credit assignment | None (each example is answered independently) | Central problem: attribute a delayed reward back to the actions that caused it (value functions / advantage) |
| Performance ceiling | Bounded by the quality of the demonstrations; hard to exceed the teacher | Can surpass the demonstrator and discover novel strategies |
| Core challenge | Fit + generalization; label quality | Exploration–exploitation trade-off; reward design; sample efficiency |
| Typical examples | Image classification, next-token prediction | Game playing (e.g., AlphaGo), fine-tuning a robot policy in the real world |

**They are usually used in succession, not as rivals:** a policy is first pre-trained / post-trained by Supervised Learning on large-scale data to get a reasonable initial behavior, then fine-tuned with Reinforcement Learning on the target task. (This mirrors the SFT → RLHF pipeline in LLMs.)

### Background:
Policies trained with imitation learning are known to suffer
from 
- compounding errors
- can only be as performant as the demonstration data

### Structure:
![项目截图](Images/pi0.6structure.png?raw=true)

#### Architecture
$τ = (o_0, a_0, · · · , o_T ) ∈ O × A · · · O$: a trajectory

$ρ_π(τ ) = p(o_0)\prod_{t = 0}^{T - 1}π(a_t|o_t)p(o_{t+1}|o_t, a_t)$

$R(τ) = \sum_{t = 0}^{T}r_t$
- $r_t$ reward
function

$V^π(o_t) = \mathbb{E}_{τ_{t+1:T}}:[\sum_{t = t}^{T}r_t]$ value function


$$A^\pi(\mathbf{o}_t, \mathbf{a}_t) = \mathbb{E}_{\rho_\pi(\tau)}\Big[\underbrace{\sum_{t'=t}^{t+N-1} r_{t'} + V^\pi(\mathbf{o}_{t+N})}_{\text{After }N\text{ steps}}\Big] - V^\pi(\mathbf{o}_t)$$
- $V^\pi(\mathbf{o}_t)$ **baseline**
- perform $\mathbf{a}_t$、sum $N$ steps rewards
- show how well the action is comared to the baseline

minimization problem:
$min_θ \mathbb{E}_{s∼ρ_{π_{ref}}}
[KL(\hatπ, π_θ)]$  

##### RECAP Architecture
![项目截图](Images/pi0.6RECAP.png?raw=true)
#### Training
- collecting data through autonomous rollouts (with optional
corrective interventions from an expert), 
- training a value function , and training a policy 

### Future Work:
- the system is not fully autonomous: it relies on human
labeling and effort for reward feedback, interventions, and
episode resets
- more sophisticated exploration methods
- extending the approach into a fully concurrent online RL
framework is a promising direction for future work

## $π_{0.7}$: a Steerable Generalist Robotic Foundation

### Highlight
Model with Emergent Capabilities
- use
diverse context conditioning during training
- additional multimodal
information that also describes the manner or strategy in which
it should do it
  - task performance
  - subgoal images

### Background
- Generalist robot manipulation policies
  - memory
  - hierarchy for long-horizon planning 
  - goal image conditioning
- Generalization across tasks and embodiments
  - leveraging human video data
  - directly leveraging Internet pre-trained foundation models during training or inference
  - improving cross-embodiment transfer between robots
  - proposed specialized hand-held
devices that can be used to collect data
- Prompting robots with subgoal images
  - allow the model to be prompted using goal images

### Structure
![项目截图](Images/pi0.7structure.png?raw=true)
#### Training
starting from a pre-trained vision language model (VLM) backbone
  - dataset $\mathcal{D}$ contains robot trajectories: observations $o_t$ and actions $a_t$
  - VLM: knowledge insulation(KI) training recipe
    - VLM backbone is supervised
with FAST tokens
    - gradients from the action expert **do not** flow into the VLM backbone
    - training example for the VLA is accompanied by a prompt or context, denoted with $\mathcal{C}_t$

##### prompt $\mathcal{C}_t$
During training, randomly **dropped out**, which provides π0.7 with the flexibility to use any
subset of the prompt components at test time
- Subtask instructions
  - During inference, $\hat{ℓ}_t$ may be produced by a learned high-level policy or a human
- Subgoal images
  - multi-view subgoals $g_t$
  - produced by lightweight world model $g_ψ$
    - trained with the objective$max_ψE_{D_g} [\mathcal{L}_{CFM} (g_t^⋆, g_ψ(o_t, \hat{ℓ}_t, m))]$
    - $\mathcal{L}_{CFM}$ standard flow matching loss
  - The image frames at the end of the segments serve as the ground-truth subgoal
  - refresh the subgoal images whenever the semantic intent changes, or after $∆$ = 4 seconds
- Episode metadata
  - to leverage lower quality demonstrations (including failures) and even autonomous data from prior models
  - metadata $m$
- Control mode 
  - for the low level action execution
  - include both joint level and end-effector actions

##### Data set
make heavy use of suboptimal robot data in training

#### Architecture
majot modifications: 
- history vision encoder from MEM
- visual subgoal images in the context

employ a block-causal masking scheme
- observation tokens and the subgoal image tokens use bidirectional attention
- following text tokens use causal attention
- embeds the state using a linear projection that
maps the state dimension to the backbone dimension
- action expert 50 tokens attend bidirectionally to each other

### Experimental Evaluation
- Out-of-the-box performance on challenging tasks
- Instruction following
  - can follow instructions that go against dataset biases
- Cross-embodiment transfer
  - successful transfer often requires the policy to discover new manipulation strategies suited to the target morphology rather than simply replicate the source behavior
- Compositional task generalization
  - can be coached to perform new longer horizon tasks purely with language
- Can π0.7 learn effectively from diverse and mixed-quality data

### Future Work:
determining what is
truly novel becomes difficult, and the model may well be
achieving generalization primarily by “remixing” skills and
behaviors from other situations
- dataset contains so many different
scenes and behaviors that potentially related skills may well
be present elsewhere in the data


## Development of $\pi$ series

| Model | Core Innovation | Progress |
|---|---|---|
| $π_0$ | VLM backbone + **flow matching** for continuous actions; separate lightweight **action expert**; 50-step action chunks | Generalist policy across diverse robots/tasks; dexterous manipulation and language following out of the box |
| $π_{0.5}$ | **Open-world generalization** via diverse co-training (web data, cross-embodiment) + hybrid multi-modal examples (subtasks, detections, actions); high-level subtask inference + low-level control | Generalizes to entirely **new homes**; generalization scales with number of training environments; outperforms $π_0$ |
| $π^*_{0.6}$ | Learns from **experience via RL** (RECAP): advantage-conditioned policy trained on autonomous rollouts + expert corrections, with a learned value function | Breaks the imitation-learning ceiling: **surpasses demonstration quality**, keeps improving through real-world deployment |
| $π_{0.7}$ | **Steerable** via diverse context conditioning (instructions, subgoal images from a world model, episode metadata, control mode); knowledge-insulation training; leverages **suboptimal / mixed-quality data** | Emergent capabilities: follows counter-bias instructions, **cross-embodiment transfer**, compositional generalization to new long-horizon tasks |

**Series evolution:** scale up data & generalization ($π_0$ → $π_{0.5}$) → improve beyond demonstrations via RL ($π^*_{0.6}$) → steerability & emergent skills from heterogeneous data ($π_{0.7}$).
