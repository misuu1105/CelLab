---
name: project-description-and-formal-instructions
description: Context, architectural rules, deliverables, and reference cellular automata specifications for the College AI Project CelLab. Use when the user mentions CelLab, asks to design, code, debug, or plan deliverables (L3, L6, L11, L13, L14) for their AI class cellular automata project, or asks to implement natural language parsing, state simulation, analytical queries, GUI, or NN/RL experiments for cellular automata.
---
# CelLab Project Guide

Provides the complete project specification, architectural constraints, milestone deliverables, and reference cellular automata models for the College AI course project **CelLab**.

## When to Use

- Working on any part of the **CelLab** project or AI class laboratory deliverables (L3, L6, L11, L13, L14)
- Designing or coding the cellular automata simulation engine, state representation, or rule engine
- Implementing natural language parsing to convert text descriptions into cellular automata rules (without runtime LLM calls)
- Implementing the 5 analytical queries (state after N steps, stability, configuration occurrence, cell counts, cycle detection)
- Building the GUI or designing the Neural Network (NN) / Reinforcement Learning (RL) dataset and experiments

## Critical Constraints & Rules

1. **Code in English**: Write all code, variable names, comments, and documentation in English unless explicitly requested otherwise.
2. **Save Agent Interactions**: Remind the user periodically to export/save chat logs, as full documentation of all conversational agent interactions is a mandatory submission deliverable.
3. **Target Maximum Score**: Aim to cover all 5 analytical questions, include complex automata (like Rule 30, Langton's Ant, or Ecosystem), and build a strong NN/RL experiment to compete for top project evaluation (Amazon Iasi Learn&Earn award).

## Core System Architecture

Design the system around four decoupled modules so each laboratory deliverable builds cleanly on the previous one:

### 1. Core Simulation Engine (Deliverable 2 - Lab 6)

- **World Matrix**: Support 2D grids (and 1D grids represented as 1xN or stacked 2D rows over time, e.g., Elementary CA / Rule 30). Support configurable boundary conditions (toroidal/wrap-around, fixed/dead borders, or reflective).
- **Cell States**: Support discrete states (enums/integers like `EMPTY`, `TREE`, `FIRE`), numerical values (floats/integers for Sandpile or Heat Diffusion), and agent-based state overlays (position + direction for Langton's Ant, or energy/timers for Predator-Prey).
- **Initial World Generation**: Allow both manual/preset initial states and procedural generation based on rules or probabilities.
- **Storage & Recovery**: Serialize and deserialize world configurations, rule definitions, and generation histories (e.g., JSON, SQLite, or HDF5/NPZ for fast dataset generation) with a toggle to store or discard intermediate generations.

### 2. Analytical Query Engine (Core Requirement)

Ensure the simulation engine can answer all 5 project questions for any known automaton:

1. **State after N iterations**: Fast-forward simulation $N$ steps and return/render the matrix.
2. **Stability detection**: Detect when `generation[t] == generation[t-1]` (static equilibrium) and return the iteration $t$ (or report if not reached within `max_steps`).
3. **Configuration occurrence**: Check whether a user-specified sub-grid pattern or full-grid state appears at any point up to $N$ iterations.
4. **Cell type count after N iterations**: Count occurrences of state $X$ at step $N$.
5. **Cycle detection**: Hash each generation state (e.g., Zobrist hashing or canonical tuple/byte hash) in a lookup table to detect if `state[t] == state[t - k]` for $k > 1$. Return the cycle start step and period $k$.

### 3. Dataset & NN/RL Experiment Module (Deliverable 3 - Lab 11)

- Generate structured datasets of `(initial_state, rule_id, step_t) -> step_t+1` or `(initial_state) -> (reaches_stability, cycle_length, final_count)`.
- Valid NN/RL experiment tracks:
  - **Neural Network**: Train a CNN/MLP/GNN to approximate state transitions, predict macro-properties (e.g., whether a configuration stabilizes or enters a cycle after $N$ steps), or classify which rule generated a given state trajectory.
  - **Reinforcement Learning**: Train an RL agent to place initial cells or perturb states to maximize/minimize a target cell type after $N$ steps (e.g., containing a Forest Fire or Epidemic, or maximizing Grass/Rabbit survival).

### 4. Natural Language Rule Parser & GUI (Deliverable 4 - Lab 13)

- **NL Parser**: Validate and translate structured or semi-structured natural language descriptions of grid dimensions, cell states, neighborhood types (Moore 8-neighbor, Von Neumann 4-neighbor, 1D 3-neighbor), and transition conditions into executable rule objects. Use a combination of tokenization, dependency/grammar parsing (e.g., Lark/ANTLR/spaCy), or intent/slot classification—never external LLM APIs.
- **Graphical User Interface**: Provide inputs for NL description, simulation parameters ($N$ generations, grid size, initial state setup, toggle for storing intermediate generations), step-by-step playback controls, and query answers.

## Deliverable Schedule

- **L3 (Deliverable 1)**: Team identification & discussion of technical stack and architecture.
- **L6 (Deliverable 2)**: Modeling an automaton (world state, rules, parameters, state storage/recovery). Tested on 2-3 different automata.
- **L11 (Deliverable 3)**: Building a dataset of generations and running a Neural Network (NN) or Reinforcement Learning (RL) experiment.
- **L13 (Deliverable 4)**: Parsing and validating natural language descriptions of automata.
- **L14 (Final)**: Complete presentation, source code, saved agent chat logs, training Q&A sets, and installation/usage docs.

## Reference Automata Catalog (15 Official Examples)

When designing the rule DSL and parser, ensure the engine can express these 15 reference models:

1. **Elementary CA (e.g., Rule 30)**: 1D binary (`0`=dead, `1`=alive). Neighborhood: `(left, current, right)`. Rule 30 maps `100, 011, 010, 001 -> 1`, others `-> 0`.
2. **Brian's Brain**: 2D, 3 states (`OFF`, `FIRING`, `DYING`). `FIRING -> DYING`, `DYING -> OFF`, `OFF -> FIRING` if `count(FIRING neighbors) == 2`.
3. **Cyclic CA**: 2D, states `0..K-1` (e.g., `K=4`). State `x` transitions to `(x + 1) mod K` if at least $T$ neighbors (e.g., 2) are in state `(x + 1) mod K`.
4. **Langton's Ant**: 2D (`WHITE`, `BLACK`) + Ant `(x, y, dir)`. On `WHITE`: turn right, flip to `BLACK`, move forward. On `BLACK`: turn left, flip to `WHITE`, move forward.
5. **Forest Fire**: 2D (`EMPTY`, `TREE`, `FIRE`). `FIRE -> EMPTY`, `TREE -> FIRE` if $\ge 1$ neighbor is `FIRE` (or spontaneously with prob $q$), `EMPTY -> TREE` with prob $p$.
6. **Sandpile Model**: 2D integer grains. If `cell >= 4`, topple: `cell -= 4`, and `north, south, east, west += 1` until stable.
7. **Traffic Simulation (Nagel-Schreckenberg / 1D)**: 1D circular road (`CAR`, `EMPTY`, velocity `0..5`). Accelerate, slow down if car ahead, random brake with prob $p$, move.
8. **Predator-Prey Grid**: 2D (`EMPTY`, `RABBIT`, `FOX`). Rabbits move randomly and reproduce after $N$ turns. Foxes eat adjacent rabbits (`+energy`), move randomly otherwise (`-1 energy`), die at `0` energy.
9. **Epidemic Simulation (SIR)**: 2D (`S`, `I`, `R`). `S -> I` if $\ge 2$ neighbors are `I` (deterministic) or with prob $p$ per `I` neighbor (stochastic). `I -> R` after $N$ turns.
10. **Schelling Segregation**: 2D (`A`, `B`, `EMPTY`). Agent is happy if $\ge X\%$ (e.g., 40%) of occupied neighbors match its group; if unhappy, moves to an `EMPTY` cell.
11. **Voter Model**: 2D (`A`, `B`). Cell adopts strict majority opinion of neighbors; ties remain unchanged.
12. **Crystal Growth**: 2D (`EMPTY`, `CRYSTAL`). Center starts with 1 `CRYSTAL`. `EMPTY -> CRYSTAL` if neighbor condition met (e.g., `count(CRYSTAL) == 1`).
13. **Diffusion / Heat Simulation**: 2D continuous/numerical. `new_value = average(cell, north, south, east, west)`.
14. **Falling Sand**: 2D directional physics (`EMPTY`, `SAND`, `ROCK`, `WATER`). Sand falls down, then down-left, then down-right. Water falls down, then flows left/right.
15. **Artificial Ecosystem**: 2D (`EMPTY`, `GRASS`, `RABBIT`, `FOX`) combining growth conditions, energy thresholds, seeking behavior, and reproduction on random `GRASS` cells.

## Gotchas

- **Stochastic vs. Deterministic Cycle/Stability Detection**: Models with probability (Forest Fire with $p, q$, probabilistic SIR, random movement in Predator-Prey) will rarely enter strict deterministic state cycles unless a seeded PRNG state is part of the state hash or the user switches them to deterministic mode. Always support a fixed random seed for reproducibility.
- **Rule Paradigms Differ**: Not all 15 examples are pure synchronous Moore/Von Neumann lookup tables. Group the engine support into: (a) Synchronous cell-state transitions (Life, Brian's Brain, Cyclic, Forest Fire, SIR, Voter, Crystal), (b) Numerical/Toppling grids (Sandpile, Heat Diffusion), and (c) Agent/Movement grids (Langton's Ant, Traffic, Schelling, Falling Sand, Predator-Prey/Ecosystem).
