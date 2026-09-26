## Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing

### Highlights
Asynchronous planner-executor modeling
- A low-frequency video planner computes reusable long-horizon world context 
- a high-frequency action expert predicts short closed-loop action chunks

Observation-Guided Video-Context Routing (OVCR)
- The latest observation routes and updates the planner context before each action prediction
- avoiding expensive video-DiT reruns at every control step

Horizon-adaptive offset training
- The action executor is trained under variable planner-executor phase offsets
- making the asynchronous interface robust to streaming deployment

Practical release surface
- RoboTwin-format data pipelines
- DeepSpeed training launchersr
- RoboTwin evaluation entrypoints
- TCP/ROS deployment templates.

### Focus:
1. 模型的输入和输出是什么: $π_θ(A_t|O^v_{≤t}, s_t, l).$
   - Input: 
      - visual observation history $O_t^v$
      - a proprioceptive state $s_t$
      - a language instruction $l$
   - Output:
     -  predicts an executable action chunk $A_t = {a_t, ..., a_{t+ha−1}}$ 
        -  DiT world planner: $Z^v_{t:t+h_v}$ future video latents
        -  Action DiT executor: action chunk $A_t$
2. Video-DiT planner 和 Action-DiT executor 分别负责什么？
   - Video-DiT:
     - takes visual latent tokens as input 
     - predict future video latents over the **longer planning horizon**
     - output:
       -  $C^p_\tau = \left\{ \left( K^{p,\ell}_\tau,\; V^{p,\ell}_\tau \right) \right\}_{\ell=1}^{L}$
       -  put through OCVR
       -  $\tilde{C}^p_t = \left\{ \left( \tilde{K}^{p,\ell}_t,\; \tilde{V}^{p,\ell}_t \right)\right\}_{\ell=1}^{L}$
   - Action-DiT:
     - receives noisy action tokens and proprioceptive tokens
     - denoises the action chunk under closed-loop robot-state feedback
3. 为什么 planner 低频运行，而 executor 高频运行？
   -  planner 低频运行: near-term frame 冗余并且信息含量低, 性能浪费
   -  executor 控制关节角度, 高频输出 action chunck 快速响应, 精细控制
4. OVCR 如何利用最新观测处理 world context？
   -  using the **latest visual observation** to convert the shared planner context into a **chunk-specific context**
   -  observation-guided routing queries:
      -  attention pooling over the current visual tokens:$Z^q_t = \mathrm{Attn}\big({B}({\text{ Q个可学习 query}}), {f_v(X^v_t)}{K}, {f_v(X^v_t)}_{V}\big)$
   -  reads planner features using the routing queries and then predicts residual key–value updates:
      -  $R^\ell_t = \mathrm{Attn}\big(Z^{q,\ell}_t, K^{p,\ell}_{\tau(t)}, V^{p,\ell}_{\tau(t)}\big),\qquad (\Delta K^{p,\ell}_t, \Delta V^{p,\ell}t) =g^\ell\psi(R^\ell_t, Z^{q,\ell}_t)$
   - chunk-specific planner context is produced by a gated residual update
     - $\tilde{K}^{p,\ell}t = K^{p,\ell}{\tau(t)} + \alpha^\ell_t \Delta K^{p,\ell}_t,\qquad\tilde{V}^{p,\ell}t = V^{p,\ell}{\tau(t)} + \alpha^\ell_t \Delta V^{p,\ell}_t$
     - get $\tilde{C}^p_t$
5. AHA-WAM 相比同步方法提升了什么？
   - 在模拟环境中提升任务完成质量
   - 在真机测试中保持任务完成质量
     - 在泛化能力上不及$\pi_{0.5}$
   - 缩短推理时长,提升动作控制频率
### Structure:
![项目截图](Images/AHA_WAM.png?raw=true)
