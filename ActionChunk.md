# Action Chunck
## Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
### Highlights
a low-cost system that performs
end-to-end imitation learning directly from real demonstrations

novel algorithm: Action Chunking with Transformers (ACT), which learns a generative
model over action sequences
- helps tackle temporally correlated confounders
- tackle pauses hard to model with Markovian single-step policies

### Background

**Imitation learning for robotic manipulation**
- Behavioral cloning (BC): simplest imitation learning — supervised learning from observations to actions
- BC improvements: history with various architectures / different training objectives / regularization
- Other directions: multi-task & few-shot, language leverage, task-structure exploitation
- Scaling with more data → generalization to new objects, instructions, scenes
- This work: low-cost yet capable system for delicate fine manipulation — hardware (teleoperation) + software (novel algorithm drastically improving previous methods)

**Addressing compounding errors**
- Major BC shortcoming: errors accumulate → drift off training distribution → hard-to-recover states; particularly prominent in fine manipulation
- DAgger & variants: on-policy interactions + expert corrections; but expert annotation time-consuming, unnatural with teleoperation interface
- Noise injection at demo collection → corrective behavior; but direct task failure in fine manipulation
- Offline synthetic correction data: limited to low-dimensional states or specific task types (e.g. grasping)
- ACT's different angle, compatible with high-dimensional visual observations: action chunking (action sequence instead of single action) reduces effective horizon; ensemble across overlapping chunks → accurate & smooth trajectories

**Bimanual manipulation**
- Long history; popularity rising as hardware costs lower
- Early: classical control with known environment dynamics — time-consuming design, inaccurate for complex physical properties
- Learning-based: RL / imitation from human demos / key points chaining motor primitives
- Fine-grained tasks (knot untying, cloth flattening, needle threading) — but on considerably more expensive robots (da Vinci, ABB YuMi)
- This work: low-cost arms (~$5k each) for high-precision, closed-loop tasks
- Teleoperation most similar to Kim et al. (joint-space mapping) — no special encoders / sensors / machined components; off-the-shelf + 3D printed parts, non-expert assembly < 2 hours

### Hardware: ALOHA: A LOW-COST OPEN-SOURCE HARDWARE
Teleoperation system
- two sets of low-cost, off-the-shelf robot arms
- joint space mapping for teleoperation
- 3D printed “see-through” fingers and fit it with gripping tape

### Structure
![项目截图](Images/ACTstructure.png?raw=true)

#### Architecture
Action Chunking: individual actions are grouped
together and executed as one unit

Temporal Ensemble: query the policy at every timestep 
- making different action chunks overlap with each other
- avoid jerky robot motion

the policy model: $\pi_\theta(a_{t:t+k} \mid s_t)$

conditional variational autoencoder (CVAE): generate an action sequence conditioned on current observations

implement the CVAE encoder and decoder with transformers:
[DetailsOfCVAE](CVAE.md)(Genertaed By GLM5.3)

**ACT Inference**
1. Given: trained policy $\pi_\theta$, episode length $T$, weight $m$.
2. Initialize FIFO buffers $B[0 : T]$, where $B[t]$ stores actions predicted for timestep $t$.
3. **for** timestep $t = 1, 2, \dots T$ **do**
4. Predict $\hat{a}_{t:t+k}$ with $\pi_\theta(\hat{a}_{t:t+k} \mid o_t, z)$ where $z = 0$   deterministic: z set to prior mode, not sampled
5. Add $\hat{a}_{t:t+k}$ to buffers $B[t : t+k]$ respectively   chunks overlap since policy queried at every timestep
6. Obtain current step actions $A_t = B[t]$
7. Apply $a_t = \sum_i w_i A_t[i] \,/\, \sum_i w_i$, with $w_i = \exp(-m \cdot i)$   temporal ensemble: exponential-weighted average over overlapping predictions

#### Training
collect human
demonstrations using ALOHA

observations are composed of 
- the current joint positions of follower robots and 
- the image feed from 4 cameras

target:$\min_θ −\sum_{s_t,a_{t:t+k∈D}}logπ_θ(at:t+k|st),$

**ACT Training**
1. Given: Demo dataset $\mathcal{D}$, chunk size $k$, weight $\beta$.
2. Let $a_t, o_t$ represent action and observation at timestep $t$, $\bar{o}_t$ represent $o_t$ without image observations.
3. **Initialize encoder $q_\phi(z \mid a_{t:t+k}, \bar{o}_t)$**   encode output z contain style information
4. Initialize decoder $\pi_\theta(\hat{a}_{t:t+k} \mid o_t, z)$
5. **for** iteration $n = 1, 2, \dots$ **do**
6. Sample $o_t, a_{t:t+k}$ from $\mathcal{D}$
7. Sample $z$ from $q_\phi(z \mid a_{t:t+k}, \bar{o}_t)$
8. Predict $\hat{a}_{t:t+k}$ from $\pi_\theta(\hat{a}_{t:t+k} \mid o_t, z)$
9. $L_{\text{reconst}} = \mathrm{MSE}(\hat{a}_{t:t+k}, a_{t:t+k})$
10. $L_{\text{reg}} = D_{KL}\big(q_\phi(z \mid a_{t:t+k}, \bar{o}_t) \,\|\, \mathcal{N}(0, I)\big)$
11. Update $\theta, \phi$ with ADAM and $L = L_{\text{reconst}} + \beta L_{\text{reg}}$

### Experimental Evaluation
- ACT significantly
outperforms previous methods that only predict single-step
actions
- more chunking and a lower effective horizon
generally improve performance
- CVAE objective is crucial when learning
from human demonstrations

### Future Work:

Hardware (ALOHA)
- multi-finger bimanual tasks (e.g. child-proof pill bottle) 
- high-force tasks (sealed bottles, heavy objects — motor torque insufficient) - fingernail tasks (tape edge, soda cans)

Policy learning (ACT) — 2 failed tasks
- unwrapping candies (unwrap 0/10): wrapper seam hard to perceive even for humans → policy peels where no seam
- opening small ziploc bag: mid-air steps fail — small pick-up differences → large bag-deformation differences → pulling region shifts
- both attributed to perception difficulty + lack of data

Directions
- pretraining 
- more data 
- better perception

## piSeries
[piSeriesNotes](piSeries.md)