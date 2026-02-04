# QAOA with Tensor-Network Simulation in CUDA-Q

This repository contains my research code and experimental results studying the
simulation of the Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut
using CUDA-Q, with a particular focus on **tensor-network (MPS) backends** and
their scaling behaviour relative to state-vector simulation.

The purpose of this project is to present a controlled set of experiments
that investigate:

- how QAOA performance scales with problem size and circuit depth,
- how tensor-network simulation compares to state-vector simulation in practice,
- what accuracy–runtime trade-offs arise when using MPS backends for QAOA.

All experiments are run by me using a fixed experimental design, and the code is
structured to make the methodology explicit and reproducible.

---

## Research Question

**When does tensor-network simulation become a practical advantage for simulating
QAOA circuits, and what is the resulting impact on solution quality for Max-Cut?**

More concretely, this project examines:
- wall-clock runtime scaling with number of qubits,
- optimizer behaviour (number of objective evaluations),
- achieved Max-Cut values relative to classical cut values,
- sensitivity to circuit depth and graph structure.

---

## Experimental Setup

### Problem
- **Problem:** Max-Cut
- **Graph families:**
  - Erdős–Rényi graphs \(G(n, p)\)
  - Random \(k\)-regular graphs
- **Graph sizes:** swept over increasing \(n\)
- **Objective Hamiltonian:**
  \[
  H = \sum_{(i,j)\in E} \tfrac{1}{2}(Z_i Z_j - 1)
  \]

### Algorithm
- **Algorithm:** QAOA
- **Depth:** fixed \(p\) per experiment
- **Ansatz:** correct Max-Cut cost unitary applied on *every edge* at each layer
- **Parameters:** \((\gamma_0,\dots,\gamma_{p-1},\beta_0,\dots,\beta_{p-1})\)

### Optimisation
- **Optimizer:** COBYLA
- **Initialisation:** random (seeded per instance)
- **Objective:** expectation value \(\langle H \rangle\) computed via `cudaq.observe`

### Simulation Backends
- `qpp-cpu` (state-vector simulation)
- `tensornet-mps` (tensor-network / MPS simulation)

---

## What This Repository Contains

qaoa_research/
├── src/
│ ├── graphs.py # graph generation + classical cut evaluation
│ ├── qaoa.py # QAOA ansatz + Hamiltonian construction
│ └── bench.py # experiment runner + metric logging
├── results/ # CSV outputs from experiments
└── notebooks/
└── analysis.ipynb # analysis and figures used to interpret results


- All **experiments** are run via `src/bench.py`.
- All **figures and conclusions** are produced in `notebooks/analysis.ipynb`.
- The notebook does not run simulations; it only consumes recorded results.

---

## Metrics Reported

For each graph instance and backend, the following quantities are recorded:

- number of nodes \(n\)
- circuit depth \(p\)
- backend type
- wall-clock optimisation time
- number of objective evaluations
- final expectation value
- best sampled bitstring
- classical Max-Cut value of that bitstring
- approximation ratio (where applicable)

These metrics allow both **performance** and **solution quality** to be assessed.

---

## Interpretation Scope

This project does **not** claim quantum advantage.

Instead, it provides:
- empirical evidence on where tensor-network simulation becomes competitive,
- concrete examples of QAOA simulation bottlenecks,
- a realistic assessment of how far classical simulation can be pushed using
  modern tensor-network backends.

---

## Notes

- All results are produced on simulators, not quantum hardware.
- Continuous improvements (e.g. bond-dimension sweeps, alternative optimizers)
  are deliberately left out to keep the experimental scope controlled.
