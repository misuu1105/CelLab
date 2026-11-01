---
name: "cellab-development-plan"
description: "Natural Language-Driven Cellular Automata Engine & AI-First Analytics Suite specification and development plan."
---

* **Team Members:** Acasandrei Mihai & Danila Emanuel
* **Language:** Python 3.11+
* **Core Ground-Truth Engine:** NumPy & SciPy
* **NLP Module:** Standalone Classical/Local NLP Pipeline (Lark / spaCy / Local Classifier) + Pydantic *(Zero Runtime External LLM Calls)*
* **GUI Framework:** Streamlit
* **AI / ML Framework:** PyTorch & scikit-learn

---

## 1. EXECUTIVE OVERVIEW & TECHNICAL STACK

Project CelLab is a discrete dynamical system and Artificial Intelligence platform designed to parse natural language rules into executable automata, generate large-scale simulation trajectories, and answer all five core analytical questions (Q1–Q5) using specialized Machine Learning models. 

To satisfy strict AI course constraints:
1. **AI-First Question Answering with Deterministic Oracle Verification:** The deterministic NumPy + Hash-Map engine (Lab 6) acts as the **Ground-Truth Oracle & Automated Dataset Generator**. In Lab 11, specialized PyTorch models are trained on these trajectories to serve as the primary AI inference engine answering Q1–Q5, while the deterministic engine provides live ground-truth benchmarking to demonstrate neural speedup vs. Stephen Wolfram's concept of computational irreducibility.

| Layer | Technology | Role & Implementation Details |
| :--- | :--- | :--- |
| **NLP Parsing** | Laya
| **AI Inference Suite** | PyTorch & scikit-learn | Primary solver for Q1–Q5 using rule-conditioned neural architectures (Neural CA U-Net, ResNet-18, Siamese CNN, DeepSets/MLP) conditioned on a rule embedding vector $R$. |
| **Ground-Truth Core** | NumPy & SciPy | Vectorized 2D/1D grid updates and matrix convolutions (`scipy.signal.convolve2d`); generates training datasets and verifies AI predictions. |
| **Oracle Fast-Forward** | Python `dict` (Hash Map) | Computes bit-packed/SHA-256 state hashes for $O(1)$ cycle detection and modular arithmetic lookups ($T + ((N - T) \bmod L)$) to label ground-truth datasets. |
| **GUI Interface** | Streamlit | Interactive web interface featuring live grid rendering, side-by-side AI vs. Exact Oracle playback, parameter tuning, and ML performance telemetry. |

---

## 2. RULE-CONDITIONED AI FORMULATION & THE 5 CORE QUESTIONS

Because users can define arbitrary rules via Natural Language, a neural network looking only at initial grid $S_0$ cannot know the transition dynamics. Therefore, every AI model in CelLab takes a **Conditioned Input Tuple**:
$$\text{Input} = \left( S_{0:k},\; R,\; N \right)$$
where $S_{0:k}$ is a short $k$-step warmup trajectory ($k \in \{1..4\}$), $R \in \mathbb{R}^d$ is the **Rule Embedding Vector** extracted directly from the validated Pydantic rule schema (encoding neighborhood weights, birth/survival bitmasks, state count, and boundary type), and $N$ is the target step horizon.

### Q1: What is the state after $N$ iterations?
* **AI Solution (Multi-Horizon Neural CA Surrogate — Conditioned U-Net / FNO):**
  * **Formulation:** Structured grid-to-grid spatial prediction $\hat{S}_N = f_\theta(S_{0:k}, R, \Delta t)$ outputting per-cell state logits $[C \times H \times W]$.
  * **Multi-Scale Stepping:** Trained with a power-of-two jump curriculum ($\Delta t \in \{1, 2, 4, 8, 16\}$) so reaching step $N$ requires $O(\log N)$ neural forward passes instead of $N$ sequential steps, mitigating autoregressive error accumulation.
* **Deterministic Ground-Truth Oracle (Lab 6 Labeler & Benchmark):**
  * Executes vectorized updates $S_{t+1} = f(S_t)$ while logging bit-packed state hashes in `hash_map[state_hash] = t`.
  * Upon cycle closure at $t_{\text{current}}$ matching $t_{\text{first}}$, computes transient $T = t_{\text{first}}$ and period $L = t_{\text{current}} - t_{\text{first}}$, retrieving exact state $S_N$ in $O(1)$ via:
    $$\text{Target Step} = T + ((N - T) \bmod L)$$

### Q2: When does the system reach a stable state?
* **AI Solution (Dual-Head Spatio-Temporal ResNet-18):**
  * **Formulation:** Takes $(S_{0:k}, R)$ and predicts via two heads:
    1. **Classification Head (Binary Cross-Entropy):** Probability $p_{\text{stable}} \in [0, 1]$ that the trajectory reaches static equilibrium ($L = 1$) within the safety horizon.
    2. **Regression Head (Huber Loss):** Predicted log-convergence step $\hat{y} = \log(1 + t_{\text{stable}})$, inverted at inference via $\hat{t}_{\text{stable}} = \lfloor \exp(\hat{y}) - 1 \rfloor$.
* **Deterministic Ground-Truth Oracle:**
  * Checks `numpy.array_equal(S_t, S_{t-1})` at each step $t$. Halts immediately upon equality ($L=1$) and logs $t$ as the ground-truth label.

### Q3: Does a particular configuration ever occur?
* **AI Solution (Siamese Cross-Attention Reachability Network):**
  * **Formulation:** Evaluates whether target sub-pattern $P$ (of shape $h \times w$) or full grid state ever appears within $N$ steps from $S_0$ under rule $R$.
  * **Architecture:** Branch A encodes $(S_{0:k}, R)$ via a 2D CNN backbone into spatial feature map $F_S$; Branch B encodes the target pattern $P$ into query embedding $F_P$. A spatial cross-attention layer computes reachability similarity scores, outputting binary reachability probability $p_{\text{occur}} \in [0, 1]$ and estimated first-occurrence step $\hat{t}_{\text{hit}}$.
* **Deterministic Ground-Truth Oracle:**
  * **Full Grid:** $O(1)$ hash comparison against target state hash.
  * **Sub-Matrix Pattern:** 2D sliding-window convolution via `scipy.signal.convolve2d` at each step until detected (`True`) or a full cycle $T + L$ completes (`False`).

### Q4: How many cells of state $X$ exist after $N$ iterations?
* **AI Solution (Population Dynamics MLP / DeepSets Regressor + U-Net Consistency):**
  * **Formulation:** Predicts exact state-distribution count vector $\hat{\mathbf{c}}_N = [\hat{c}_0, \hat{c}_1, \dots, \hat{c}_{C-1}]$ at step $N$ using a dual approach:
    1. **Macro Population Regressor (MLP):** Takes initial spatial statistics (per-state densities, spatial entropy, cluster sizes, rule vector $R$, and $\log N$) and outputs predicted cell counts via Poisson / MSE loss.
    2. **Micro Surrogate Aggregation:** Sums thresholded softmax probabilities from Q1's Neural CA output $\sum_{i,j} \mathbb{I}(\arg\max \hat{S}_N(i,j) == X)$ and ensembles both predictions.
* **Deterministic Ground-Truth Oracle:**
  * Retrieves $S_N$ via the Fast-Forwarding Engine and runs `numpy.count_nonzero(S_N == X)` in a single vectorized operation.

### Q5: Does the system enter a cycle, and what are its transient $T$ and period $L$?
* **AI Solution (Attractor & Wolfram Class Classifier + Periodicity Regressor):**
  * **Formulation:** Multi-task CNN backbone over $(S_{0:k}, R)$ with three outputs:
    1. **Cycle Existence & Dynamical Regime Head:** Classifies trajectory into Fixed Point ($L=1$), Periodic Attractor ($L > 1$), or Chaotic/Aperiodic (Wolfram Classes I–IV).
    2. **Transient Regressor:** Predicts $\log(1 + \hat{T})$ (steps before entering the cycle).
    3. **Period Regressor:** Predicts $\log(1 + \hat{L})$ (cycle length $k > 1$).
* **Deterministic Ground-Truth Oracle:**
  * Maintains bit-packed byte hashes in `dict` (`state_hash -> step_index`). On collision, confirms cycle (`True`) and records exact $(T, L)$.

---

## 3. EQUAL WORK DIVISION MATRIX

| Milestone / Deliverable | Acasandrei Mihai (Core Engine, Math & AI/ML Models) | Danila Emanuel (NLP Parser, GUI & System Integration) |
| :--- | :--- | :--- |
| **Lab 3: Proposal (Deliverable 1)** | Technical architecture for NumPy ground-truth engine, Hash-Map fast-forwarding math, and AI model formulations for Q1–Q5. | Standalone NLP grammar/schema specifications, Pydantic rule validation models, Rule Embedding $R$ spec, and Streamlit UI wireframes. |
| **Lab 6: Core Engine (Deliverable 2)** | Implements 1D/2D grid transition logic, bit-packed state hashing, $O(1)$ `dict` cycle detection, modular $N$-step lookup, and automated Q1–Q5 ground-truth labeler. | Implements Streamlit interactive grid viewer, simulation playback controls, boundary/parameter toggles, and JSON/NPZ state serialization & recovery. |
| **Lab 11: Dataset & AI Suite (Deliverable 3)** | Builds automated trajectory dataset generator; implements and trains PyTorch models answering Q1 (U-Net), Q2 (ResNet), Q3 (Siamese CNN), Q4 (MLP), and Q5 (Attractor Net). | Integrates PyTorch inference pipeline into Streamlit UI; builds side-by-side AI vs. Exact Oracle visual comparison, error heatmaps, and Q1–Q5 telemetry panels. |
| **Lab 13: NL Parser (Deliverable 4)** | Maps validated Pydantic rule schemas into executable NumPy transition kernels and deterministic Rule Embedding vectors $R$ for neural conditioning. | Implements standalone local NL parser (regex/Lark grammar/local classifier -> Pydantic), rule validation error handling, and interactive NL rule entry UI. |
| **Lab 14: Final Defense** | Co-presents AI vs. Oracle accuracy/speedup benchmarks, computational irreducibility analysis, and Q1–Q5 neural metrics. | Demonstrates live Streamlit application, natural language rule parsing, saved development agent chat logs, and system Q&A. |

---

## 4. SEMESTER LABORATORY SCHEDULE & DELIVERABLES

* **Lab 3 (Deliverable 1) — Technical Solution & Team Roles:** Submit system architecture, AI-first Q1–Q5 formulation, standalone NLP -> Pydantic schema design, and work division matrix.
* **Lab 6 (Deliverable 2) — Automata Modeling & Ground-Truth Engine:** Deliver working Streamlit GUI + NumPy Simulation Engine tested across multiple automata paradigms (e.g., Game of Life, Rule 30, Brian's Brain, Langton's Ant, Forest Fire) with state storage/recovery and $O(1)$ cycle/fast-forward oracle.
* **Lab 11 (Deliverable 3) — Dataset Generation & AI/ML Experiments:** Export labeled simulation datasets `(S_0:k, R, N) -> (Q1..Q5 labels)`, train the PyTorch model suite to answer all 5 questions, and benchmark AI inference time and accuracy against the exact engine in the GUI.
* **Lab 13 (Deliverable 4) — Standalone Natural Language Rule Parser:** Fully integrate the offline NL parser and Pydantic validator to translate free-form/semi-structured text rules into both executable simulations and neural rule embeddings.
* **Lab 14 (Final Defense):** Present complete application, benchmark reports, training Q&A sets, installation documentation, and exported conversational agent development logs.