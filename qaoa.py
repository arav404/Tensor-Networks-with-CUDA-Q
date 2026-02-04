"""Construction of QAOA circuits and evaluation using CUDA‑Q.

This module encapsulates the logic for building and evaluating QAOA
ansätze for Max‑Cut instances using CUDA‑Q.  It exposes helper functions
to build the corresponding cost Hamiltonian, construct a specialised
kernel for a given edge list, evaluate expectations and sample bitstrings.

The QAOA ansatz implemented here uses the standard alternation of
problem and mixer unitaries with a set of variational parameters
``gamma`` and ``beta``.  For a circuit depth ``p`` there are ``2*p``
parameters arranged as ``[gamma[0], …, gamma[p−1], beta[0], …, beta[p−1]]``.

The cost unitary is implemented correctly for arbitrary graphs by
applying an entangling gate sequence to each edge in the graph on every
layer.  The sequence ``CX–RZ(2*gamma)–CX`` realises ``exp(-i γ Z_i Z_j)`` up to
global phases.  For the mixer unitary we apply an ``RX(2*beta)`` rotation on
every qubit.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np

try:
    import cudaq
    from cudaq import spin
except ImportError as exc:  # pragma: no cover
    # If cudaq is not available the module can still be imported for
    # inspection, but any call that attempts to use it will raise.  This
    # allows static analysis and documentation generation without a CUDA‑Q
    # installation.
    cudaq = None  # type: ignore
    spin = None   # type: ignore
    _import_error = exc


def build_hamiltonian(edge_list: Iterable[Tuple[int, int]]) -> 'cudaq.SpinOperator':
    """Construct the Max‑Cut cost Hamiltonian for a graph.

    The Hamiltonian for Max‑Cut is

        H = ∑_{(i,j)∈E} ½ (Z_i Z_j − 1)

    where the constant shift makes the minimum eigenvalue correspond to
    the negative of the cut value.  When passed to ``cudaq.observe`` this
    shift ensures that minimising the expectation value corresponds to
    maximising the cut size.

    Parameters
    ----------
    edge_list : Iterable[Tuple[int,int]]
        A list of undirected edges of the graph.

    Returns
    -------
    cudaq.SpinOperator
        The corresponding spin operator.

    Raises
    ------
    RuntimeError
        If CUDA‑Q is not available.
    """
    if spin is None:
        raise RuntimeError(
            "cudaq is not installed; cannot construct spin Hamiltonians")
    hamiltonian: cudaq.SpinOperator = 0
    for u, v in edge_list:
        hamiltonian += 0.5 * (spin.z(u) * spin.z(v) - 1)
    return hamiltonian


def get_qaoa_kernel(edge_list: List[Tuple[int, int]]) -> Callable:
    """Return a CUDA‑Q kernel for QAOA on a specific graph.

    This function returns a specialised kernel that closes over the edge
    list.  The returned kernel accepts three arguments: ``qubit_count``
    (int), ``p`` (int) and ``params`` (a flat list of length ``2*p``).
    Within each layer it applies the cost unitary for every edge, then
    applies the mixer unitary to every qubit.  The equal superposition
    state is prepared up front.

    Parameters
    ----------
    edge_list : List[Tuple[int,int]]
        List of edges defining the graph.

    Returns
    -------
    Callable
        A function decorated with ``@cudaq.kernel`` which implements the
        QAOA ansatz for the given edge set.
    """
    if cudaq is None:
        raise RuntimeError(
            "cudaq is not installed; cannot build QAOA kernels")

    # Copy edges locally to avoid capturing the outer variable by reference
    edges = list(edge_list)

    @cudaq.kernel
    def kernel(qubit_count: int, p: int, params: List[float]):  # type: ignore
        qvec = cudaq.qvector(qubit_count)
        # Prepare equal superposition
        h(qvec)
        # Apply p layers
        for layer in range(p):
            gamma = params[layer]
            beta = params[p + layer]
            # Cost unitary: apply Z_i Z_j interactions for each edge
            for (u, v) in edges:
                # CX–RZ–CX implements ZZ interaction up to global phase
                x.ctrl(qvec[u], qvec[v])
                rz(2.0 * gamma, qvec[v])
                x.ctrl(qvec[u], qvec[v])
            # Mixer unitary
            for qubit in range(qubit_count):
                rx(2.0 * beta, qvec[qubit])
    return kernel


def expectation_value(params: Iterable[float], hamiltonian: 'cudaq.SpinOperator',
                      qubit_count: int, p: int, kernel: Callable) -> float:
    """Compute the expectation value of the cost Hamiltonian.

    Parameters
    ----------
    params : Iterable[float]
        Flat list of length ``2*p`` containing the QAOA parameters.
    hamiltonian : cudaq.SpinOperator
        The cost Hamiltonian constructed with :func:`build_hamiltonian`.
    qubit_count : int
        Number of qubits (equal to number of graph vertices).
    p : int
        Depth of the QAOA circuit.
    kernel : Callable
        A specialised QAOA kernel returned by :func:`get_qaoa_kernel`.

    Returns
    -------
    float
        Expectation value of the Hamiltonian.
    """
    if cudaq is None:
        raise RuntimeError("cudaq is not installed; cannot evaluate expectation")
    # Convert params to a Python list in case an ndarray is passed
    param_list = list(params)
    result = cudaq.observe(kernel, hamiltonian, qubit_count, p, param_list)
    return result.expectation()


def sample_bitstrings(params: Iterable[float], qubit_count: int, p: int,
                       kernel: Callable, shots: int = 1024) -> 'cudaq.SampleResult':
    """Sample bitstrings from the QAOA ansatz using the given parameters.

    Parameters
    ----------
    params : Iterable[float]
        Flat list of length ``2*p`` containing the QAOA parameters.
    qubit_count : int
        Number of qubits.
    p : int
        Depth of the QAOA circuit.
    kernel : Callable
        A specialised QAOA kernel as returned by :func:`get_qaoa_kernel`.
    shots : int, optional
        Number of samples to draw, by default 1024.

    Returns
    -------
    cudaq.SampleResult
        A CUDA‑Q sample result object from which bitstrings and counts can
        be extracted.  See the CUDA‑Q documentation for details.
    """
    if cudaq is None:
        raise RuntimeError("cudaq is not installed; cannot sample bitstrings")
    param_list = list(params)
    return cudaq.sample(kernel, qubit_count, p, param_list, shots=shots)