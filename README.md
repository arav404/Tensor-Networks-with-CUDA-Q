# CUDA-Q QAOA Benchmarking Framework

This repository contains a modular research codebase for studying the performance of
Quantum Approximate Optimization Algorithm (QAOA) implementations using
[CUDA‑Q](https://nvidia.github.io/cuda-quantum/).  The goal of this project
is to facilitate reproducible experiments on Max‑Cut instances and to make it
easy to swap graph families, circuit depths, back‑end simulators and
optimisation strategies.  The code is structured as a small Python package in
the `src/` directory along with an example analysis notebook under
`notebooks/`.

Unlike earlier prototyping notebooks, this project:

* **Implements the cost unitary correctly for arbitrary graphs.**  Each QAOA
  layer applies an `exp(-i γ Z_i Z_j)` interaction for every edge in the
  target graph rather than restricting interactions to nearest neighbours.
* **Separates problem generation, circuit construction and benchmarking.**  The
  `graphs.py` module provides functions to generate graphs and evaluate cut
  values, `qaoa.py` encapsulates construction of the QAOA ansatz and
  expectation calculation using CUDA‑Q, and `bench.py` orchestrates the
  optimisation and timing loops.
* **Records timing and quality metrics.**  The benchmarking harness measures
  wall‑clock times for optimisation, collects the best expectation values,
  samples from the optimised circuit and computes classical cut values and
  approximation ratios.  Results are returned as Pandas dataframes which can
  be saved to CSV for later analysis.
* **Supports multiple graph families and back‑ends.**  At present Erdos–Rényi
  (`G(n,p)`) and k‑regular random graphs are implemented, but additional
  families can be added easily.  Back‑end selection is handled via
  `cudaq.set_target(...)`; both state‑vector (`qpp-cpu`) and tensor network
  (`tensornet-mps`) simulators are supported.

## Installation

This project depends on CUDA‑Q, igraph and matplotlib.  A typical
`requirements.txt` is provided.  Because CUDA‑Q packages are not available on
the default Python Package Index, you will need to follow NVIDIA’s
installation instructions to obtain a working CUDA‑Q installation.  On a
machine with CUDA‑capable GPUs and a working C++ toolchain you can run

```sh
pip install cuda-quantum --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
```

Note: The code in this repository will import `cudaq`.  If you cannot install
CUDA‑Q on your system you can still inspect the source files, but running
benchmarks will fail.

## Structure

```
qaoa_research/
├── README.md          # this document
├── requirements.txt   # Python dependencies
├── src/               # Python package with modular implementation
│   ├── __init__.py
│   ├── graphs.py      # graph generation and cut evaluation
│   ├── qaoa.py        # QAOA circuit and expectation evaluation
│   └── bench.py       # benchmarking harness
├── results/           # directory for CSV results (created at runtime)
└── notebooks/
    └── analysis.ipynb # example analysis (to be filled in by the user)
```

### `src/graphs.py`

This module provides utilities to generate random graphs and to compute
classical cut values.  It wraps the [`igraph`](https://igraph.org/) library
and defines functions such as:

* `generate_k_regular(n: int, k: int, seed: int) -> ig.Graph`: generate a
  random *k*‑regular graph on *n* vertices using the given seed.
* `generate_erdos(n: int, p: float, seed: int) -> ig.Graph`: generate a
  random Erdős–Rényi graph with edge probability *p*.
* `edges(graph) -> List[Tuple[int,int]]`: return a list of undirected edges.
* `cut_value(bitstring: str, edges: List[Tuple[int,int]]) -> int`:
  compute the number of edges crossing the cut induced by the bitstring.

These functions isolate the graph‑generation logic from the quantum code and
make it easy to experiment with other graph families.

### `src/qaoa.py`

This module encapsulates construction of the QAOA ansatz and evaluation
functions using CUDA‑Q.  Key components include:

* `build_hamiltonian(edges: List[Tuple[int,int]]) -> cudaq.SpinOperator`:
  construct the Max‑Cut Hamiltonian for the given edge list.  The
  Hamiltonian takes the form
  \(H = \sum_{(i,j)∈E} \tfrac{1}{2}(Z_i Z_j - 1)\).
* `qaoa_kernel(qubit_count: int, p: int, parameters: List[float], edges: List[Tuple[int,int]])`:
  a CUDA‑Q kernel decorated with `@cudaq.kernel` that prepares the
  equal‑superposition state, applies the cost unitary for every edge in
  every layer, and applies the mixer unitaries.  The kernel expects a
  parameter vector of length `2*p` where the first *p* entries are the
  `γ` values and the next *p* entries are the `β` values.
* `expectation_value(params, hamiltonian, qubit_count, p, edges) -> float`:
  evaluate the expectation value of the cost Hamiltonian with respect to the
  QAOA state for the given parameters using `cudaq.observe`.
* `sample_bitstrings(params, qubit_count, p, edges, shots: int)`:
  sample bitstrings from the optimised circuit.

Having a separate module makes it easy to switch between back‑ends and to
instrument the simulation calls for timing.

### `src/bench.py`

The `bench.py` module orchestrates optimisation and benchmarking.  It exposes
functions such as:

* `run_instance(graph: ig.Graph, p: int, optimizer, backend: str)`:
  given a single graph instance, depth `p` and a CUDA‑Q optimiser,
  construct the Hamiltonian, minimise its expectation value using the
  QAOA kernel, record timing information, sample an approximate
  solution and compute its classical cut value.
* `run_experiment(graph_family: str, sizes: List[int], p: int, reps: int, backend: str, **backend_options)`:
  iterate over node counts and random seeds, generate graphs of the
  specified family, call `run_instance` and aggregate the results into
  a pandas `DataFrame`.

The benchmark functions do not plot by themselves – they simply return
structured data.  Users can call these functions from a notebook or a
script, save the resulting dataframes to `CSV` under `results/`, and
generate plots at their discretion.

## Getting Started

1. **Install dependencies.**  Ensure you have CUDA‑Q and igraph installed as
   described above.
2. **Run a quick test:**

   ```python
   from qaoa_research.src.graphs import generate_k_regular
   from qaoa_research.src.bench import run_instance
   import cudaq
   import pandas as pd

   # pick a backend
   cudaq.set_target('qpp-cpu')

   # generate a simple graph
   g = generate_k_regular(4, 3, seed=0)

   # optimise with 1 layer
   result = run_instance(g, p=1, optimizer=cudaq.optimizers.COBYLA(), backend='qpp-cpu')
   print(result)
   ```

3. **Run a full experiment:**  Use `run_experiment` to sweep over node
   sizes and repetitions.  For example:

   ```python
   from qaoa_research.src.bench import run_experiment
   import pandas as pd

   data = run_experiment(graph_family='k_regular', sizes=[6,8,10], p=2, reps=3, backend='qpp-cpu', k=3)
   data.to_csv('results/cpu_kregular_p2.csv', index=False)
   ```

4. **Analyse results:**  The provided notebook under `notebooks/` offers a
   starting point for loading CSV files, computing statistics and
   visualising scaling behaviour.  Feel free to adapt it for your own
   experiments.

## Acknowledgements

This codebase was refactored from an experimental notebook exploring
tensor‑network simulators for QAOA.  It incorporates suggestions for
improving methodological rigour including correctly implementing the cost
unitary, isolating graph generation, using rolling seeds, and capturing
timing and quality metrics.  Further enhancements such as support for
rolling hedge ratios, advanced optimisers or hardware back‑ends are left
for future work.