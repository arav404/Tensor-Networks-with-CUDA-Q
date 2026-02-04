"""Graph generation and classical cut evaluation.

This module isolates all of the classical graph functions used in the QAOA
benchmark.  By keeping graph generation separate from the quantum code, it
becomes straightforward to swap in different problem families or to perform
classical analysis of instances.  Functions in this file operate on
`igraph.Graph` objects and return simple Python data structures such as
lists of edges or integer cut values.  No CUDA‑Q code appears here.
"""

from __future__ import annotations

from typing import List, Tuple

import igraph as ig
import numpy as np

def generate_k_regular(n: int, k: int, seed: int) -> ig.Graph:
    """Return a random k‑regular graph on n vertices.

    A k‑regular graph has exactly k incident edges on every vertex.  For a
    simple undirected graph to exist we require `n * k` to be even and
    `k < n`.  The underlying `igraph.Graph.K_Regular` constructor does not
    accept a seed argument directly, so we seed both NumPy and the igraph
    PRNG before constructing the graph.  Note that the igraph random number
    generator is global; if you call this function from multiple threads
    concurrently you may need to protect the seed setting.

    Parameters
    ----------
    n : int
        The number of vertices.
    k : int
        The degree of each vertex.
    seed : int
        Seed used to initialise both NumPy and igraph random generators.

    Returns
    -------
    ig.Graph
        A random k‑regular graph of order n.
    """
    if k >= n:
        raise ValueError(f"k must be less than n (got k={k}, n={n})")
    if (n * k) % 2 != 0:
        raise ValueError(f"n * k must be even to construct a k‑regular graph (got {n}*{k})")

    # Seed both NumPy and igraph's random number generators for reproducibility.
    np.random.seed(seed)
    ig.random.seed(seed)

    return ig.Graph.K_Regular(n, k, directed=False, multiple=False)


def generate_erdos(n: int, p: float, seed: int) -> ig.Graph:
    """Return an Erdős–Rényi random graph G(n,p).

    Each possible undirected edge is included independently with probability
    `p`.  Self‑loops are disallowed.  As with `generate_k_regular`, we
    seed both NumPy and igraph before graph generation.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Probability of including each edge (0 ≤ p ≤ 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ig.Graph
        A random Erdős–Rényi graph.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0,1] (got {p})")
    np.random.seed(seed)
    ig.random.seed(seed)
    return ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)


def edges(graph: ig.Graph) -> List[Tuple[int, int]]:
    """Return a list of undirected edges from the given igraph graph.

    Each edge is represented as a tuple `(u, v)` with `u < v`.  The
    ordering is arbitrary but deterministic given the igraph internal
    ordering of edges.  Duplicates and self‑loops are not expected when
    generating simple graphs.

    Parameters
    ----------
    graph : ig.Graph
        An undirected igraph instance.

    Returns
    -------
    List[Tuple[int,int]]
        A list of edges.
    """
    return [(e.source, e.target) if e.source < e.target else (e.target, e.source)
            for e in graph.es]


def cut_value(bitstring: str | List[int], edge_list: List[Tuple[int, int]]) -> int:
    """Compute the cut size induced by a bitstring on a graph.

    Given a binary assignment to the vertices (encoded either as a string of
    '0'/'1' characters or as a list of integers 0/1), the cut size is the
    number of edges whose endpoints are assigned to different bits.  This
    corresponds to the classical Max‑Cut objective.

    Parameters
    ----------
    bitstring : str or List[int]
        A binary string or list of length equal to the number of vertices.
    edge_list : List[Tuple[int,int]]
        The edges of the graph as returned by :func:`edges`.

    Returns
    -------
    int
        The number of edges crossing the cut defined by the bitstring.
    """
    # Coerce bitstring into a list of integers 0/1.
    if isinstance(bitstring, str):
        assignment = [1 if ch == '1' else 0 for ch in bitstring]
    else:
        assignment = list(bitstring)
    cut = 0
    for u, v in edge_list:
        if assignment[u] != assignment[v]:
            cut += 1
    return cut


def brute_force_max_cut(graph: ig.Graph) -> Tuple[int, str]:
    """Compute the maximum cut of a graph via brute force.

    This function exhaustively enumerates all 2ⁿ assignments to the n
    vertices and returns the maximum cut size and a corresponding bitstring.
    It is intended for small graphs (n ≲ 14) as the cost grows
    exponentially.  This function is useful for sanity checks or for
    computing approximation ratios when benchmarking QAOA on tiny instances.

    Parameters
    ----------
    graph : ig.Graph
        The graph on which to compute the maximum cut.

    Returns
    -------
    Tuple[int, str]
        A pair `(max_cut, bitstring)` containing the optimal cut size and
        one optimal bitstring.
    """
    n = graph.vcount()
    e_list = edges(graph)
    max_cut = -1
    best = None
    for i in range(1 << n):
        bitstring = format(i, f'0{n}b')
        c = cut_value(bitstring, e_list)
        if c > max_cut:
            max_cut = c
            best = bitstring
    return max_cut, best