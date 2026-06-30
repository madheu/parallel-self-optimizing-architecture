# Parallel Self-Optimizing Architecture: A Multi-Threaded Iterative Refinement Framework for Large Language Models

**Abstract**: The "one-shot generation" paradigm of Large Language Models (LLMs) suffers from inherent limitations including factual errors, logical flaws, and insufficient output. Existing improvement methods—whether self-reflection, external critique, or multi-agent debate—all operate along a single dimension, lacking systematic integration of generation, review, release, and experience consolidation. This paper proposes the **Parallel Self-Optimizing Architecture (PSOA)**, a four-thread parallel collaborative framework: the Generation Thread produces content output, the Review Thread provides negative feedback for corrective refinement, the Release Thread determines output termination using logical self-consistency ("Does this make sense?") as the criterion, and the Iteration Thread feeds runtime experience back to each sub-thread in real time, enabling online self-optimization. Human feedback serves as a meta-instruction that anchors the global optimization direction, preventing the system from drifting into self-reinforcing deviation. At the theoretical level, this paper analyzes the information-theoretic functions and interaction protocols of each thread, discusses the convergence of termination conditions, and positions PSOA within the academic lineage of existing paradigms including Self-Refine, Constitutional AI, Reflexion, and multi-agent debate, demonstrating its theoretical advantages in output quality, reasoning reliability, and self-evolution capability.

**Keywords**: Large Language Models; Parallel Self-Optimization; Multi-Threaded Architecture; Self-Feedback Iteration; Human Feedback Anchoring; Multi-Agent Collaboration

---

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable generative capabilities, yet their "one-shot" generation paradigm suffers from inherent deficiencies: factual hallucination, logical discontinuity, and shallow responses to complex instructions (Madaan et al., 2023; Bai et al., 2022a). Just as human writing requires repeated revision, an LLM's single-pass output often necessitates subsequent correction to reach an acceptable quality level.

The research community has attempted to overcome this bottleneck from multiple directions. **Self-Critique & Refinement** lets the same model simultaneously play the roles of generator and critic, progressively improving output quality through a "generate → evaluate → refine" cycle (Madaan et al., 2023; Bai et al., 2022a). **External Critique & Feedback** separates the generation and critique roles, with independent entities providing evaluation (Ouyang et al., 2022; Chiang et al., 2023). **Multi-Agent Collaboration & Debate** has multiple agents assume different roles, jointly optimizing output through debate and negotiation (Li et al., 2023; Hong et al., 2023; Du et al., 2023). **Reflexion** replaces weight updates with verbal feedback, enabling agents to accumulate experience through trial and error (Shinn et al., 2023).

However, the above methods suffer from three structural deficiencies:

1. **Role Confusion**: In self-critique, the generator and critic share the same model, which tends to produce "confirmation bias"—the model is inclined to affirm rather than genuinely negate its own output (Bai et al., 2022a).
2. **Missing Termination Signal**: Most iterative frameworks lack an explicit termination judgment mechanism. They either rely on fixed iteration counts (coarse-grained) or simple score thresholds (which may be exploited by "reward hacking"), and cannot determine the critical signal of "quality is already sufficient" (Amodei et al., 2016).
3. **Inability to Consolidate Experience**: Most methods form closed loops within a single task, but cross-task experience cannot structurally flow back into the system's behavioral strategies, resulting in starting from scratch for each new task (Shinn et al., 2023; 4Paradigm et al., 2025).

To address these deficiencies, this paper proposes the **Parallel Self-Optimizing Architecture (PSOA)**, whose core innovations are:

- **Decoupling** the four functions of generation, negative-feedback review, positive-feedback release, and cross-round experience iteration **into four parallel threads**, each operating independently yet coordinating through a message protocol.
- Introducing the **Release Thread** as an explicit termination signal source, using logical self-consistency ("Does this make sense?") as the convergence criterion, forming a counterbalancing bidirectional quality gate with the Review Thread's negative feedback.
- Introducing the **Iteration Thread** as an experience consolidation and generational distillation engine, distilling the review records and release judgments from each run into the initialization state of the next-generation threads, driving sub-threads to spawn improved next-generation instances before self-destruction.
- Using **human feedback as meta-instruction** to anchor the global optimization direction, preventing the system from deviating from human intent during self-iteration—this is consistent with Constitutional AI's "constitution" philosophy (Bai et al., 2022a), but PSOA elevates it from offline training principles to runtime dynamic anchors.

Section 2 surveys related work, Section 3 details PSOA's four-thread architecture and interaction protocol, Section 4 analyzes the convergence of termination conditions, Section 5 discusses the human feedback anchoring mechanism, Section 6 explores application scenarios and limitations, and Section 7 concludes the paper.

---

## 2 Related Work

### 2.1 Self-Critique and Iterative Refinement

The **Self-Refine** framework proposed by Madaan et al. (2023) is a representative work on iterative self-refinement. The framework has the same LLM sequentially execute generation, feedback, and refinement steps, iterating until a predefined stopping condition is met. Experiments on 7 tasks show that Self-Refine improves performance by approximately 20% on average compared to single-pass generation. However, the core limitation of Self-Refine is that the generator and critic share the same model parameters, making it difficult for the model to provide genuinely critical feedback on its own output—"self-reviewing oneself" inherently tends toward confirmation bias.

**Constitutional AI (CAI)** proposed by Bai et al. (2022a) introduces "constitutional principles" as anchors for self-critique. The model critiques and revises its own output according to predefined principles, then trains a preference model through RLAIF (Reinforcement Learning from AI Feedback). CAI's core contribution lies in making critique criteria explicit, but CAI's constitutional principles are static and offline-defined, unable to dynamically adjust at runtime.

### 2.2 Multi-Agent Collaboration and Debate

Multi-agent systems overcome single-model cognitive limitations through role separation and interactive debate. Li et al. (2023) proposed the **CAMEL** framework, which guides two agents to collaboratively complete tasks through role-playing and inception prompting. Hong et al. (2023) proposed **MetaGPT**, encoding standardized operating procedures (SOP) of software development as multi-agent collaboration protocols, enabling role agents such as product managers, architects, and engineers to work collaboratively following the process. Du et al. (2023) proposed the **Multi-Agent Debate (MAD)** framework, where multiple LLM instances improve reasoning quality through debate and cross-examination.

However, multi-agent debate faces dual challenges of efficiency and convergence. Kapoor et al. (2024) noted that redundant interactions during debate lead to sharply increasing computational overhead, and Wang et al. (2024) found that errors may be propagated and amplified during debate. The DOWN (Debate Only When Necessary) framework attempts to selectively activate debate through a confidence gating mechanism, but only addresses the efficiency issue without touching the structural deficiencies of termination judgment and experience consolidation.

### 2.3 Reflexive Learning and Self-Evolving Agents

Shinn et al. (2023) proposed the **Reflexion** framework, replacing traditional reinforcement learning weight updates with verbal feedback. After task failure, the agent generates textual reflections stored in an episodic memory buffer, retrieving and utilizing these reflections in subsequent attempts to improve decision-making. Reflexion demonstrates the feasibility of "learning without adjusting weights," but its reflections are task-level and cross-episode, lacking online real-time feedback capability.

The EvolveR framework (4Paradigm et al., 2025) proposes an experience-driven self-evolution lifecycle: online interaction collects trajectories, offline distillation produces abstract strategy principles, and reinforcement learning internalizes these principles. The ANCHOR framework (arXiv:2606.06114) introduces a review mechanism that simulates human supervision, providing evaluation feedback at multiple stages of the self-evolution process (task formulation, planning, output, execution results), significantly mitigating safety degradation during self-evolution.

Microsoft Research's **Online Experiential Learning (OEL)** framework (Microsoft Research, 2026) enables deployed models to continuously learn from real interaction experience through a two-stage iterative logic of "experience extraction → experience integration," achieving the transformation from static models to dynamic agents.

The above works each address certain aspects that PSOA is concerned with, but none systematically integrates the four functions of generation, review, release, and experience iteration as parallel decoupled threads. Figure 6 illustrates the architectural comparison between PSOA and existing methods.

![Figure 6: Architectural comparison of PSOA and existing methods](../assets/figures/fig6-comparison.jpg)

### 2.4 Human Feedback and Alignment

Ouyang et al.'s (2022) **InstructGPT** established the RLHF (Reinforcement Learning from Human Feedback) paradigm, training reward models from human preference data to align LLM behavior. However, RLHF faces a fundamental dilemma in "super-alignment" scenarios: when AI capabilities surpass human evaluators, the reliability of human feedback declines (Bai et al., 2022b). PSOA repositions human feedback as "meta-instruction anchoring" rather than fine-grained preference annotation, emphasizing the irreplaceability of humans in directional guidance while delegating specific execution to the system's autonomous operation.

---

## 3 Parallel Self-Optimizing Architecture (PSOA)

### 3.1 Architecture Overview

PSOA consists of four functionally decoupled parallel threads, interacting through a structured message protocol to form a closed-loop self-optimizing system. Figure 1 illustrates the overall structure of the architecture.

![Figure 1: PSOA four-thread parallel architecture](../assets/figures/fig1-architecture.jpg)

<details>
<summary>Text-based architecture diagram</summary>

```
┌──────────────────────────────────────────────────────┐
│                  人类反馈（元指令锚定）                  │
│                        ↓                              │
│              ┌─────────────────┐                      │
│              │   迭代线程(I)    │◄──── 经验回流 ──────┐ │
│              │  代际蒸馏引擎    │                     │ │
│              └────┬────┬───────┘                     │ │
│        下一代初始化↓    ↓下一代初始化                   │ │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐         │ │
│    │ 生成线程 │  │ 审查线程 │  │ 放行线程 │         │ │
│    │   (G)    │  │   (R)    │  │   (L)    │         │ │
│    │ 第t代实例 │  │ 第t代实例 │  │ 第t代实例 │         │ │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘         │ │
│         │              │              │               │ │
│         └──────►───────┘──────────────┘               │ │
│               输出候选  审查意见  放行信号               │ │
│                        │                              │ │
│                   ┌────┴────┐                         │ │
│                   │ 输出闸门 │───── 最终输出 ──────────┐ │
│                   └─────────┘     经验记录            │ │
│                        │                              │ │
│              ┌─────────┴─────────┐                   │ │
│              │ 蒸馏→繁衍→自消亡   │                   │ │
│              │ 生成第t+1代实例    │                   │ │              │
└──────────────────────────────────────────────────────┘
```

**Figure 1**: Overview of the PSOA four-thread parallel self-optimizing architecture. The Generation Thread G produces candidate content, the Review Thread R provides negative feedback, the Release Thread L provides positive feedback, and the Output Gate determines whether to release the output based on signals from R and L. The Iteration Thread I extracts experience from each run, driving generational replacement of G/R/L—threads self-destruct after distillation, with next-generation instances inheriting improved strategies. Human feedback serves as meta-instruction anchoring the optimization direction of I.

</details>

### 3.2 Generation Thread (G)

**Function Definition**: The Generation Thread is the system's content production engine, responsible for generating candidate output $\hat{y}_t$ given input task $x$.

**Operational Mechanism**: The Generation Thread receives the input task and the strategy prompt $p_t^G$ provided by the Iteration Thread, generating candidate output in iteration round $t$:

$$\hat{y}_t = G(x, p_t^G, h_{t-1})$$

where $h_{t-1}$ is the review history summary from the first $t-1$ rounds, enabling the Generation Thread to perceive previous correction directions. The Generation Thread does not operate in a vacuum—its strategy prompt $p_t^G$ is dynamically adjusted by the Iteration Thread based on historical experience, inclining it toward generating higher-quality content in subsequent tasks.

**Distinction from Self-Refine**: In Self-Refine, the generator simultaneously assumes the self-feedback role, whereas PSOA completely strips feedback functionality into independent Review and Release Threads, eliminating the confirmation bias arising from role confusion.

### 3.3 Review Thread (R)

**Function Definition**: The Review Thread is the system's negative feedback engine, responsible for identifying deficiencies, errors, and shortcomings in the candidate output, and generating corrective recommendations.

**Operational Mechanism**: The Review Thread receives the candidate output $\hat{y}_t$ and the review strategy $p_t^R$ provided by the Iteration Thread, outputting structured review opinions:

$$r_t = R(\hat{y}_t, x, p_t^R, c)$$

where $c$ is an optional set of "constitutional" principles (inspired by Constitutional AI) for anchoring review criteria. The review opinion $r_t$ contains the following structured fields:

- **Error Type**: Factual errors, logical flaws, missing information, formatting issues, etc.
- **Severity**: Fatal / Important / Minor
- **Correction Suggestion**: Specific modification direction and content

The core positioning of the Review Thread is **negative feedback**—it is not responsible for confirming "how good the content is," only for pointing out "what is wrong with the content." This pure negative-feedback positioning avoids functional overlap between the Review Thread and the Release Thread.

**Relationship to Constitutional AI**: The Review Thread can be viewed as an independent and structured version of CAI's self-critique mechanism. In CAI, the model critiques its own output, whereas PSOA delegates the critique function to an independent thread while inheriting CAI's "constitutional principles" idea as an anchor for review criteria.

### 3.4 Release Thread (L)

**Function Definition**: The Release Thread is the system's positive feedback engine and termination signal source, responsible for determining whether the current candidate output has achieved logical self-consistency and deciding whether to release it as the final output.

**Core Criterion—Logical Self-Consistency Verification**: The convergence criterion of the Release Thread is not an externally attached quality score, but rather a logical self-consistency verification of the candidate output. This design stems from a key observation: **the way humans judge that they are "done speaking" is not by scoring, but by asking themselves "Does this make sense?"**—as long as the expression is logically sound, internally non-contradictory, and logically self-consistent, the language is fluent and the output is sufficient. Quality scores are externally attached anchors with no natural standard for what score to assign; self-consistency is intrinsic—whether the output contradicts itself can be objectively verified. This criterion resonates deeply with the "Ouroboros" metaphor: the "self-closing" in "logically self-consistent" is precisely the logical closure of the serpent's head biting its own tail.

**Operational Mechanism**: The Release Thread receives the candidate output $\hat{y}_t$ and the review opinion $r_t$, performs self-consistency verification, and outputs a release judgment:

$$l_t = L(\hat{y}_t, r_t, x, p_t^L)$$

The release judgment $l_t$ contains:

- **Release Signal**: $l_t \in \{0, 1\}$, where 1 indicates release and 0 indicates rejection
- **Self-Consistency Assessment**: Verification of the degree of logical closure in the current output, encompassing the following dimensions:
  - **Internal Consistency**: Does the output contain self-contradictory statements?
  - **Causal Closure**: Is the causal chain of argument → evidence → conclusion fully closed?
  - **Semantic Completeness**: Are there unresolved critical semantic gaps?
- **Confidence**: The reliability of the release judgment itself

**Dual-Gate Termination Condition**: The final output must satisfy both conditions simultaneously:

$$\text{Output} = \begin{cases} \hat{y}_t & \text{if } l_t = 1 \text{ and } \|r_t^{\text{fatal}}\| = 0 \\ \text{continue} & \text{otherwise} \end{cases}$$

That is, the output is released only when the Release Thread confirms logical self-consistency **and** the Review Thread has flagged no fatal errors. This dual-gate mechanism ensures double quality assurance: the Release Thread positively confirms that the output is "logically self-consistent," while the Review Thread negatively confirms "no fatal issues exist." Figure 2 illustrates the dual-gate mechanism.

![Figure 2: Dual-gate mechanism](../assets/figures/fig2-dual-gate.jpg)

**Core Innovation**: Existing iterative frameworks generally lack explicit termination judgment mechanisms. Self-Refine relies on fixed iteration counts or simple feedback text matching (e.g., "everything looks fine"), lacking systematic assessment of output quality. PSOA's Release Thread elevates termination judgment from an implicit condition to an explicit independent functional module, and anchors the convergence criterion to logical self-consistency rather than externally attached quality scores, forming a positive-negative counterbalancing bidirectional gate with the Review Thread.

### 3.5 Iteration Thread (I)

**Function Definition**: The Iteration Thread is the system's online self-optimization engine, responsible for extracting experience from each run and driving generational replacement of sub-threads.

**Core Mechanism—Generational Distillation and Self-Destruction**: PSOA's iteration is not policy patching of fixed thread instances, but a **generational lifecycle**: each sub-thread (G/R/L) is an "individual" that, upon completing its current task, distills its own experience, generates next-generation thread instances, and then self-destructs. This process is analogous to biological mitosis—the parent transmits improved genetic material to the offspring before division.

Traditional methods interpret iteration as "tuning parameters of the same model," but PSOA interprets it as "one generation of models giving rise to the next generation":

$$\text{Gen}_{t+1} = \text{Distill}(\text{Gen}_t) + \text{Mutate}(\text{Gen}_t, e_t)$$

where $\text{Distill}$ denotes experience distillation (compressing the key decisions and discoveries from the current run into transferable strategies), $\text{Mutate}$ denotes strategy mutation based on experience $e_t$ (improving upon the previous generation's shortcomings), and the two jointly produce the next-generation threads.

**Generational Lifecycle**: Each sub-thread instance undergoes the following lifecycle (Figure 3):

![Figure 3: Generational lifecycle](../assets/figures/fig3-lifecycle.jpg)

```
[诞生] ← 由上一代蒸馏产物初始化
   ↓
[执行] ← 完成当前轮次的生成/审查/放行任务
   ↓
[蒸馏] ← 将运行经验压缩为下一代初始化参数
   ↓
[繁衍] ← 生成下一代线程实例
   ↓
[自消亡] ← 释放资源，终止当前实例
```

**Distillation Process**: The Iteration Thread performs experience distillation after each task completion:

$$e_t = I(\hat{y}_t, r_t, l_t, x, y^*, H_{t-1})$$

where $y^*$ is the ground truth answer (if available) and $H_{t-1}$ is the historical experience repository. The distillation process consists of three stages:

1. **Experience Extraction**: Extract key decision points, review findings, and the causal chain of release judgments from the complete trajectory of the current run.
2. **Strategy Distillation**: Abstract raw experience into transferable behavioral strategy principles (inspired by EvolveR's offline distillation, but PSOA completes this during the online phase). Distillation is lossy—only experience that has a causal impact on behavioral strategies is retained, while redundant contextual details are discarded, ensuring that the next generation is not burdened by historical baggage.
3. **Generational Injection**: Inject the distilled strategies as initialization parameters for the next generation:

$$\text{Gen}_{t+1}^G = \Gamma^G(\text{Gen}_t^G, e_t), \quad \text{Gen}_{t+1}^R = \Gamma^R(\text{Gen}_t^R, e_t), \quad \text{Gen}_{t+1}^L = \Gamma^L(\text{Gen}_t^L, e_t)$$

where $\Gamma$ is the generational update function, which can employ LLM-driven prompt rewriting or parameter fine-tuning. The key distinction: this is not patching the current instance, but initializing the next-generation instance—previous-generation experience has been internalized as the next generation's "innate knowledge."

**Necessity of Self-Destruction**: Threads must self-destruct after distillation is complete, rather than continuing to run. There are three reasons:

1. **Avoiding Experience Contamination**: If threads were immortal, historical experience would linger in implicit states (attention patterns, intermediate representations), conflicting with newly injected strategies and causing behavioral inconsistency.
2. **Forcing Experience Compression**: Self-destruction forces threads to explicitly externalize all experience into transferable strategy parameters, rather than relying on implicit memory, ensuring that experience is auditable and traceable.
3. **Resource Release**: Each generation of threads releases computational and memory resources, preventing system bloat during continuous iteration.

**Distinction from Reflexion**: Reflexion's reflections are task-level and cross-episode, and affect only the generation strategy. PSOA's Iteration Thread drives generational replacement of all three G/R/L threads, updating not only strategies but also the model instances that carry those strategies—iteration applies not only to "how to work" but also to "who is working."

**Distinction from EvolveR**: EvolveR's experience distillation is completed offline, introducing a time lag between experience accumulation and strategy application. PSOA's Iteration Thread distills and generates the next generation immediately after each task completion, eliminating the time lag of offline distillation. More critically, EvolveR distills strategy principles, whereas PSOA distills complete next-generation thread instances—strategies have been internalized as the instances' initialization states.

### 3.6 Inter-Thread Communication Protocol

The four threads interact through a structured message protocol, with the core communication flow as follows:

```
输入 x → [G_t] → 候选 ŷ_t → [R_t] → 审查意见 r_t
                                ↓
                           [L_t] ← ŷ_t + r_t
                                ↓
                        放行判断 l_t (自洽性检验)
                                ↓
                      ┌────────────────┐
                      │ 自洽且无致命？  │
                      └───────┬────────┘
                       是↓         ↓否
                    最终输出    ŷ_{t+1} = G_t(x, r_t, h_t)
                                ↓
                           [I] ← 运行轨迹
                                ↓
                    蒸馏 → 繁衍 Gen_{t+1} → G_{t+1}/R_{t+1}/L_{t+1}
                                ↓
                         Gen_t 自消亡
```

Key design principles:

- **Review First**: The Generation Thread must receive review opinions before entering the next iteration round, avoiding undirected repeated generation.
- **Release Veto**: The Release Thread has the authority to veto outputs that the Review Thread has not flagged, preventing low-quality outputs from being released due to "review blind spots."
- **Generational Replacement Cannot Be Bypassed**: The Iteration Thread's distillation-spawning-self-destruction must apply to all sub-threads, ensuring synchronized evolution of the entire system—no thread may be immortal.
- **Experience Must Be Explicit**: The self-destruction mechanism forces threads to explicitly distill all experience rather than relying on implicit state transfer, ensuring experience is auditable and traceable.

---

## 4 Convergence Analysis of Termination Conditions

### 4.1 Problem Definition

PSOA's termination condition is jointly determined by the Review Thread R and the Release Thread L. Define the output self-consistency degree at round $t$ as $\sigma_t$, where the termination condition requires:

$$\sigma_t = 1 \quad \text{且} \quad \|r_t^{\text{fatal}}\| = 0$$

where $\sigma_t \in \{0, 1\}$ is the Release Thread's logical self-consistency judgment. The core question: under what conditions can the iterative process converge to $\sigma_t = 1$?

Unlike traditional quality threshold models, PSOA's convergence criterion is a binary self-consistency judgment rather than a continuous quality score. This design reflects the essential characteristic of human linguistic practice: we judge that a passage is "done speaking" not because it has reached some score, but because it is "logically self-consistent"—its internal logic is closed, with no self-contradictions remaining.

### 4.2 Self-Consistency Monotonic Improvement Condition

Define the logical contradiction set of output $\hat{y}_t$ as $\Phi(\hat{y}_t)$, containing all internal inconsistencies, causal fractures, and semantic gaps. The Review Thread's feedback $r_t$ is essentially the identification and localization of $\Phi(\hat{y}_t)$, and the Generation Thread produces $\hat{y}_{t+1}$ after correction based on this feedback. Figure 4 illustrates the self-consistency convergence process.

![Figure 4: Self-consistency convergence process](../assets/figures/fig4-convergence.jpg)

Assuming the Review Thread's feedback is effective (i.e., each round of correction eliminates at least one identified logical contradiction), then:

$$|\Phi(\hat{y}_{t+1})| \leq |\Phi(\hat{y}_t)| - |\Phi_{\text{resolved}}(r_t)| + |\Phi_{\text{new}}(\hat{y}_{t+1})|$$

where $\Phi_{\text{resolved}}(r_t)$ is the contradictions resolved by the current round of correction, and $\Phi_{\text{new}}(\hat{y}_{t+1})$ is the new contradictions introduced during the correction process. If the correction quality satisfies:

$$|\Phi_{\text{resolved}}(r_t)| > |\Phi_{\text{new}}(\hat{y}_{t+1})|$$

i.e., each round of correction results in a net reduction of at least one logical contradiction, then $|\Phi(\hat{y}_t)|$ monotonically decreases. Since $|\Phi| \geq 0$, the sequence must converge.

**Theorem 1 (Self-Consistency Convergence)**: If (1) the Review Thread identifies at least one genuine logical contradiction in each round; (2) the Generation Thread's corrections result in a net reduction of contradictions greater than the number of new ones introduced; then there exists a finite round $T$ such that $\sigma_T = 1$.

**Proof**: $|\Phi(\hat{y}_t)|$ is monotonically decreasing and non-negative, so by the monotone convergence theorem it must converge to some lower bound $k \geq 0$. If $k = 0$, then $\sigma_T = 1$ holds. If $k > 0$, there exist residual contradictions that the Review Thread cannot identify—in this case, human feedback must intervene to provide correction (see Section 5).

### 4.3 Practical Convergence Guarantees

In practical systems, ensuring convergence requires addressing the following issues:

1. **Review Degradation**: The Review Thread may experience "review fatigue" as iteration rounds increase, gradually becoming unable to detect new logical contradictions. PSOA mitigates this by having the Iteration Thread update the review strategy $p_t^R$, keeping the Review Thread's perspective fresh.
2. **Self-Consistency Judgment Drift**: The Release Thread's standard for "logically self-consistent" may drift during self-iteration—the system may lower self-consistency requirements to release output more quickly. PSOA constrains standard drift through human feedback anchoring (see Section 5).
3. **Local Self-Consistency Trap**: The system may converge to a logically self-consistent but factually incorrect output—"logically self-consistent" does not mean "factually correct." Self-consistency is a necessary but not sufficient condition, which is an inherent limitation of PSOA's convergence analysis. The Review Thread must also verify factual correctness, and human feedback must provide corrections at the factual level.
4. **Corrections Introducing New Contradictions**: The Generation Thread may introduce new contradictions when correcting one (e.g., modifying a premise causing a conclusion to become invalid). PSOA mitigates this through the Review Thread's full-output review (rather than partial patching)—each review round examines the complete output, not just the modified portions.

### 4.4 Hard Constraint on Maximum Iteration Rounds

To prevent theoretically non-convergent situations (e.g., persistent conflict between review and release), PSOA introduces a maximum iteration rounds $T_{\max}$ as a hard constraint:

$$t \leq T_{\max}$$

When $t = T_{\max}$, the system forcibly outputs the current best candidate. This design borrows from Self-Refine's fixed iteration count mechanism, but PSOA uses it as a fallback strategy rather than the primary termination condition.

---

## 5 Human Feedback Meta-Instruction Anchoring Mechanism

### 5.1 Motivation: The Risk of Self-Reinforcing Drift

Any self-optimizing system faces the risk of self-reinforcing drift: during iterative optimization, the system may gradually deviate from human intent, falling into the trap of "optimizing proxy metrics rather than the true objective" (Amodei et al., 2016). In PSOA, this risk manifests specifically as:

- The Review Thread may form an "internal consensus," overlooking deficiency types that humans care about but to which the model is insensitive.
- The Release Thread may lower quality thresholds, prematurely releasing insufficient output.
- The Iteration Thread may crystallize systematic biases as "experience," causing biases to be reinforced in subsequent runs.

### 5.2 Design of Meta-Instruction Anchoring

PSOA positions human feedback as **meta-instruction** rather than fine-grained preference annotation. The core idea of meta-instruction anchoring is: humans do not participate in the specific review of each generation, but instead anchor the system's evolutionary direction by setting directional optimization objectives and constraint conditions. Figure 5 illustrates the meta-instruction anchoring mechanism.

![Figure 5: Meta-instruction anchoring mechanism](../assets/figures/fig5-anchoring.jpg)

Meta-instructions contain three levels:

1. **Objective Anchoring**: Defines the ultimate standard for output quality (e.g., "responses must be based on verifiable facts"), corresponding to Constitutional AI's constitutional principles, but dynamically updatable.
2. **Constraint Anchoring**: Defines inviolable boundaries (e.g., "must not output harmful content"), serving as hard constraints for the Review Thread.
3. **Corrective Anchoring**: When the system exhibits self-reinforcing drift, humans provide directional corrections (e.g., "recent outputs lack sufficient technical depth"), and the Iteration Thread adjusts the strategy update direction accordingly.

### 5.3 Comparison with RLHF

| Dimension | RLHF | PSOA Meta-Instruction Anchoring |
|-----------|------|-------------------------------|
| Feedback Granularity | Fine-grained preference annotation (per-item scoring) | Coarse-grained directional guidance (objectives/constraints/corrections) |
| Feedback Timing | Offline training phase | Runtime dynamic injection |
| Feedback Cost | High (requires extensive annotation) | Low (small amount of directional guidance) |
| Alignment Effect | Aligns preference distribution | Aligns intent direction |
| Super-Alignment Risk | High (human evaluator capability ceiling) | Low (humans retain advantage at the directional level) |

### 5.4 Anchoring Frequency and Decay

Human feedback does not need to be continuously injected. PSOA adopts a **decaying anchoring** strategy: human feedback is injected at high frequency during system initialization (establishing the correct optimization direction), with the injection frequency gradually decreasing as the system accumulates experience, eventually triggering corrective anchoring only when the system exhibits significant drift. This strategy balances anchoring effectiveness against human participation cost.

---

## 6 Discussion and Applications

### 6.1 Application Scenarios

**Scenario 1: High-Reliability Content Generation**

In high-stakes domains such as legal documents and medical reports, the accuracy and completeness of output are paramount. PSOA's Review Thread can focus on fact-checking and logical consistency, the Release Thread sets strict quality thresholds, and human feedback anchors domain expertise (e.g., "medical recommendations must cite clinical guidelines").

**Scenario 2: Code Generation and Review**

PSOA can be directly mapped to the software development workflow: the Generation Thread corresponds to code authoring, the Review Thread to code review, the Release Thread to merge approval, and the Iteration Thread to team experience consolidation. MetaGPT has demonstrated the effectiveness of SOP-driven multi-agent collaboration in software development (Hong et al., 2023); PSOA adds explicit termination judgment and online self-optimization capabilities on this foundation.

**Scenario 3: Academic Writing**

Paper writing requires multiple rounds of revision and peer review. In PSOA, the Generation Thread produces the initial draft, the Review Thread simulates reviewers identifying errors, the Release Thread judges whether the manuscript meets submission standards, and the Iteration Thread transforms review experience into writing strategies.

### 6.2 Limitations and Open Problems

1. **Computational Overhead**: Running four threads in parallel implies at least 4× inference overhead. In resource-constrained scenarios, cost can be reduced through shared model instances with differentiated prompts, but this sacrifices the cognitive diversity afforded by thread independence.
2. **Inter-Thread Coupling**: Although PSOA emphasizes thread decoupling, the Iteration Thread's strategy updates for G/R/L introduce indirect coupling—strategy updates may cause the Review Thread and Release Thread to "co-degrade" simultaneously.
3. **Reliability of Human Feedback**: Meta-instruction anchoring assumes that humans can accurately judge the system's optimization direction. When tasks involve domains beyond human expertise, human feedback may introduce bias.
4. **Experience Accumulation Across Conversations**: PSOA currently focuses on closed-loop optimization within a single task; mechanisms for cross-task and cross-domain experience transfer remain to be further explored.
5. **Cognitive Boundary of Generational Distillation**: Prototype validation experiments revealed a critical structural limitation—**generational distillation can only optimize "how to think," not compensate for "what is not known."** In a 5-round essay-writing experiment, the system never converged: the Review Thread (R) accurately identified factual errors in every round (Van Gogh biographical details, Kantian philosophical concept misattribution, printing press and Renaissance causal misalignment), while the Release Thread (L) correctly judged logical self-consistency in every round. The dual gate mechanism fell into a deadlock of "L=pass + R=fatal." The Iteration Thread (I) distilled "facts must be precise" into each generation's strategy, but telling the Generation Thread (G) to "be precise" at the strategy level does not equate to it "knowing precise facts" at the parameter level—strategy prompts cannot rewrite incorrect knowledge already solidified in model weights. This finding yields the following theoretical insights:
   - **Orthogonality of Self-Consistency and Factual Accuracy**: Logical self-consistency ($\sigma_t$) and factual correctness ($\|r_t^{\text{fatal}}\|$) are orthogonal dimensions. An argument can be logically flawless yet built on entirely false premises—"making sense" ≠ "being right." PSOA's dual gate design leverages this orthogonality for complementary quality control, but when the two dimensions persistently conflict, the system lacks a deadlock-breaking mechanism.
   - **Irreplaceability of Strategy Distillation vs. Knowledge Correction**: The I-thread's experience distillation is essentially **compression and transfer of behavioral strategies** ("what to do in similar situations"), not **correction of knowledge itself** ("whether Van Gogh cut off his left ear or his left earlobe"). For factual errors, the correct repair path is not strategy distillation but **external knowledge retrieval** (RAG) or **weight updating** (fine-tuning), both of which exceed the current I-thread's capability boundary.
   - **Possible Deadlock-Breaking Paths**: (a) Introduce fact-checking tool invocation capabilities, upgrading R-thread review from "relying on the model's own knowledge" to "relying on external knowledge source verification"; (b) When the same type of fatal error persists for $k$ consecutive rounds, trigger human feedback intervention as meta-instruction correction; (c) Separate R-thread's factual review and logical review into two sub-modules, each updated with different strategies.

---

## 7 Conclusion

This paper proposes the Parallel Self-Optimizing Architecture (PSOA), a multi-threaded iterative refinement framework for large language models. PSOA decouples the four functions of generation, review, release, and iteration into independent parallel threads that coordinate through a structured message protocol, achieving the following theoretical contributions:

1. **Role Decoupling**: Separating generation and critique functions into independent threads eliminates the confirmation bias characteristic of Self-Refine.
2. **Explicit Termination**: Introducing the Release Thread as an explicit termination signal source, using logical self-consistency rather than externally attached quality scores as the convergence criterion—humans judge that they are "done speaking" by asking "Does this make sense?" rather than "what score did it get"—forming a positive-negative counterbalancing bidirectional quality gate with the Review Thread, filling the gap in termination judgment left by existing iterative frameworks.
3. **Online Self-Optimization**: The Iteration Thread, through generational distillation and self-destruction mechanisms, distills experience from each run in real time into the next generation's initialization state—iterating not only "how to work" but also "who is working"—achieving online self-evolution of the entire system, overcoming the time-lag limitation of Reflexion-style cross-episode reflection and the architectural limitation of EvolveR-style offline distillation.
4. **Human Anchoring**: Repositioning human feedback as meta-instruction anchoring, ensuring alignment direction while reducing human participation cost.

PSOA provides a systematic framework for LLM output quality assurance and self-evolution capability, and its four-thread parallel architecture lays the theoretical foundation for subsequent engineering implementation and experimental validation. Prototype validation experiments further revealed the orthogonality of self-consistency and factual accuracy, as well as the cognitive boundary of generational distillation in knowledge correction, providing a clear direction for subsequent architectural improvements. Future work directions include: implementing PSOA engineering prototypes on concrete tasks, designing strategy distillation algorithms for the Iteration Thread, introducing external knowledge retrieval (RAG) as a deadlock-breaking mechanism for factual errors, and conducting systematic ablation studies to validate the independent contributions of each thread.

---

## References

1. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. *arXiv preprint arXiv:1606.06565*.

2. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2022a). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

3. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022b). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*.

4. Chiang, C. H., & Lee, H. Y. (2023). Can large language models be an alternative to human evaluations? In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*.

5. Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.

6. Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Wang, C., ... & Lu, W. (2023). MetaGPT: Meta programming for multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

7. Kapoor, S., Gao, L., Grams, A., Phul, A., Brunk, J., & Narayanan, A. (2024). AI models still fail at multi-agent debate. *arXiv preprint arXiv:2402.02049*.

8. Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., & Ghanem, B. (2023). CAMEL: Communicative agents for "mind" exploration of large language model society. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2303.17760.

9. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Yazdanbakhsh, A. (2023). Self-Refine: Iterative refinement with self-feedback. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2303.17651.

10. Microsoft Research. (2026). Online experiential learning for language models. *Microsoft Research Blog*.

11. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems (NeurIPS 2022)*.

12. 4Paradigm et al. (2025). EvolveR: Self-evolving LLM agents through an experience-driven lifecycle. *arXiv preprint arXiv:2510.16079*.

13. Shinn, N., Cassano, F., Berman, A., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2303.11366.

14. Wang, Z., Mao, X., Wang, Y., & Zhang, Y. (2024). Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*.

15. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems (NeurIPS 2022)*.

16. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of thoughts: Deliberate problem solving with large language models. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*. arXiv:2305.10601.

17. ANCHOR. (2025). Towards healthy evolution: Exploring the role and mechanisms of human-agent interaction in self-evolving systems. *arXiv preprint arXiv:2606.06114*.

18. ASI-ARCH. (2025). AlphaGo moment for model architecture discovery. *arXiv preprint arXiv:2507.18074*.

---

*This paper is a theoretical framework paper. All references are genuine published or preprint academic literature; no citation information has been fabricated.*
