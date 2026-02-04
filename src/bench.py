"""Benchmark harness for QAOA.

This module provides functions to run optimisation experiments on random
graphs using the QAOA ansatz implemented in :mod:`qaoa_research.src.qaoa`.
It is intended for programmatic use from scripts or notebooks.  Results
are returned as pandas dataframes which can then be saved to CSV or
visualised using matplotlib or seaborn.  Timing information and
call counts are recorded to support scaling analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

import time
import numpy as np
import pandas as pd

try:
    import cudaq  # type: ignore
except ImportError as exc:  # pragma: no cover
    cudaq = None  # type: ignore
    _import_error_bench = exc

from . import graphs
from . import qaoa


@dataclass
class InstanceResult:
    """Container for the result of optimising a single graph instance.

    Attributes
    ----------
    n : int
        Number of vertices in the graph.
    p : int
        Depth of the QAOA circuit.
    backend : str
        The CUDA‑Q simulation back‑end used (e.g. ``'qpp-cpu'`` or
        ``'tensornet-mps'``).
    graph_family : str
        The family of the graph (e.g. ``'k_regular'`` or ``'erdos'``).
    graph_seed : int
        Seed used to generate the random graph instance.
    graph_params : Dict[str, Any]
        Additional parameters describing the graph (e.g. ``{'k':3}`` or
        ``{'p':0.7}``).
    call_count : int
        Number of objective function evaluations performed during
        optimisation.
    runtime : float
        Total wall‑clock time spent in the optimisation loop, in seconds.
    objective_value : float
        The minimum expectation value of the Hamiltonian returned by the
        optimiser.  The corresponding Max‑Cut value is ``-objective_value``.
    bitstring : str
        The bitstring returned by sampling the optimised circuit with
        ``qaoa.sample_bitstrings`` and taking the most probable result.
    cut_value : int
        The classical cut value associated with ``bitstring``.
    initial_parameters : np.ndarray
        The initial point used by the optimiser.  Recorded for
        reproducibility.
    optimal_parameters : np.ndarray
        The optimiser’s final parameter vector.
    """

    n: int
    p: int
    backend: str
    graph_family: str
    graph_seed: int
    graph_params: Dict[str, Any]
    call_count: int
    runtime: float
    objective_value: float
    bitstring: str
    cut_value: int
    initial_parameters: np.ndarray
    optimal_parameters: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy arrays to Python lists for JSON/CSV serialisation
        d['initial_parameters'] = d['initial_parameters'].tolist()
        d['optimal_parameters'] = d['optimal_parameters'].tolist()
        return d


def run_instance(graph: graphs.ig.Graph, p: int, optimizer: Any, backend: str,
                 graph_family: str, graph_seed: int, graph_params: Dict[str, Any],
                 backend_options: Optional[Dict[str, Any]] = None) -> InstanceResult:
    """Optimise the QAOA ansatz for a single graph instance.

    This function constructs the cost Hamiltonian, builds a specialised
    QAOA kernel, and runs a classical optimiser to minimise the
    expectation value.  It measures wall‑clock time, counts objective
    evaluations, samples a candidate solution and evaluates its cut
    value.

    Parameters
    ----------
    graph : igraph.Graph
        The problem instance.
    p : int
        Depth of the QAOA circuit.
    optimizer : Any
        A CUDA‑Q optimiser instance.  The caller must configure
        ``optimizer.initial_parameters`` if desired.
    backend : str
        Name of the CUDA‑Q simulation back‑end (e.g. ``'qpp-cpu'`` or
        ``'tensornet-mps'``).
    graph_family : str
        Name of the graph family for bookkeeping.
    graph_seed : int
        Random seed used to generate the graph.
    graph_params : Dict[str,Any]
        Dictionary of parameters used in graph generation (e.g. ``{'k':3}``).
    backend_options : Optional[Dict[str,Any]], optional
        Additional keyword arguments to pass to ``cudaq.set_target``.

    Returns
    -------
    InstanceResult
        A dataclass containing optimisation results and metadata.
    """
    if cudaq is None:
        raise RuntimeError("cudaq is not installed; cannot run QAOA instances")

    # Select back‑end
    backend_options = backend_options or {}
    cudaq.set_target(backend, **backend_options)

    # Build edge list and Hamiltonian
    edge_list = graphs.edges(graph)
    hamiltonian = qaoa.build_hamiltonian(edge_list)
    qubit_count = graph.vcount()
    # Prepare specialised kernel
    kernel = qaoa.get_qaoa_kernel(edge_list)

    # Determine parameter dimension
    param_dim = 2 * p
    # If the optimiser does not have an initial point, supply a random one
    initial_parameters = getattr(optimizer, 'initial_parameters', None)
    if initial_parameters is None:
        # Draw random initial parameters in [−π/8, π/8] as is common
        initial_parameters = np.random.uniform(-np.pi/8.0, np.pi/8.0, param_dim)
        optimizer.initial_parameters = initial_parameters
    else:
        initial_parameters = np.array(initial_parameters)

    call_count = 0

    def objective(params: Iterable[float]) -> float:
        nonlocal call_count
        call_count += 1
        return qaoa.expectation_value(params, hamiltonian, qubit_count, p, kernel)

    # Run the optimiser
    start = time.perf_counter()
    objective_value, optimal_params = optimizer.optimize(dimensions=param_dim, function=objective)
    runtime = time.perf_counter() - start

    # In CUDA‑Q, ``optimizer.optimize`` returns (min_value, param_vector)
    # The Max‑Cut value is the negative of the expectation value.

    # Sample bitstrings from the optimised circuit to obtain a candidate solution
    sample_result = qaoa.sample_bitstrings(optimal_params, qubit_count, p, kernel)
    bitstring = sample_result.most_probable()
    cut_val = graphs.cut_value(bitstring, edge_list)

    return InstanceResult(
        n=qubit_count,
        p=p,
        backend=backend,
        graph_family=graph_family,
        graph_seed=graph_seed,
        graph_params=graph_params,
        call_count=call_count,
        runtime=runtime,
        objective_value=objective_value,
        bitstring=bitstring,
        cut_value=cut_val,
        initial_parameters=initial_parameters,
        optimal_parameters=np.array(optimal_params),
    )


def run_experiment(graph_family: str, sizes: Iterable[int], p: int, reps: int,
                   backend: str, optimizer_factory: Callable[[], Any],
                   seed_offset: int = 0, backend_options: Optional[Dict[str, Any]] = None,
                   **graph_kwargs: Any) -> pd.DataFrame:
    """Run a sweep of QAOA experiments over multiple graph sizes.

    This high‑level function generates random graphs of the specified
    family, runs QAOA on each instance with the requested depth and
    backend, and aggregates the results into a pandas dataframe.  The
    caller provides a factory function to create fresh optimiser objects
    for each instance to avoid cross‑contamination of state.  Graph
    generation parameters are passed via ``graph_kwargs``.

    Parameters
    ----------
    graph_family : str
        Which graph family to generate.  Supported values are ``'k_regular'``
        and ``'erdos'``.  For ``'k_regular'`` you must supply ``k`` as a
        keyword argument; for ``'erdos'`` you must supply ``p`` (edge
        probability).
    sizes : Iterable[int]
        Sequence of graph sizes (number of vertices) to test.
    p : int
        Depth of the QAOA circuit.
    reps : int
        Number of random instances (seeds) to generate for each size.
    backend : str
        Name of the CUDA‑Q back‑end to use.
    optimizer_factory : Callable[[], Any]
        Function returning a new optimiser instance.  Each call should
        produce a fresh optimiser with its own initial parameters.  For
        example, ``lambda: cudaq.optimizers.COBYLA()``.
    seed_offset : int, optional
        Base offset added to the seed for reproducibility.  The actual
        seed for the ``i``‑th repetition of size ``n`` will be
        ``seed_offset + i``.
    backend_options : Optional[Dict[str, Any]], optional
        Additional keyword arguments passed to ``cudaq.set_target``.
    **graph_kwargs : Any
        Additional keyword arguments passed to the graph generator.  For
        ``'k_regular'`` one must pass ``k=int``; for ``'erdos'`` one must
        pass ``p=float``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per graph instance containing all
        ``InstanceResult`` fields plus the Max‑Cut value (negative of
        objective) and the approximation ratio if the exact optimum is
        available for the small graphs (optional).
    """
    if graph_family not in {"k_regular", "erdos"}:
        raise ValueError(
            f"Unsupported graph family '{graph_family}'. Supported: 'k_regular', 'erdos'.")
    results: List[InstanceResult] = []
    for n in sizes:
        for rep in range(reps):
            seed = seed_offset + rep
            # Generate the graph
            if graph_family == "k_regular":
                if "k" not in graph_kwargs:
                    raise ValueError("k_regular graph requires parameter 'k'")
                k = graph_kwargs["k"]
                graph = graphs.generate_k_regular(n, k, seed)
                g_params = {"k": k}
            else:
                if "p" not in graph_kwargs:
                    raise ValueError("erdos graph requires parameter 'p'")
                prob = graph_kwargs["p"]
                graph = graphs.generate_erdos(n, prob, seed)
                g_params = {"p": prob}

            # Create a fresh optimiser instance
            opt = optimizer_factory()
            # We rely on run_instance to initialise optimiser.initial_parameters

            res = run_instance(graph, p, opt, backend,
                               graph_family=graph_family,
                               graph_seed=seed,
                               graph_params=g_params,
                               backend_options=backend_options)
            results.append(res)
    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    # Add Max‑Cut value (negative objective) as a separate column
    df["maxcut_value"] = -df["objective_value"]
    # If graphs are small, optionally compute the exact optimum for ratio
    # This is expensive; skip by default.  Users can post‑process if desired.
    return df