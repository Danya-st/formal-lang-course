import scipy.sparse as sp
from networkx import MultiDiGraph

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Reachability with regular constraints and multiple starting vertices."""
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    query_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    sources = sorted(graph_fa.start_states)
    if (
        not sources
        or not graph_fa.final_states
        or not query_fa.start_states
        or not query_fa.final_states
    ):
        return set()

    n_sources = len(sources)
    n_query = query_fa.size
    n_vertices = graph_fa.size
    shape = (n_sources * n_query, n_vertices)

    seed_rows, seed_cols = [], []
    for i, vertex in enumerate(sources):
        for q0 in query_fa.start_states:
            seed_rows.append(i * n_query + q0)
            seed_cols.append(vertex)
    frontier = sp.csr_matrix(
        ([True] * len(seed_rows), (seed_rows, seed_cols)), shape=shape, dtype=bool
    )
    visited = frontier.copy()

    identity = sp.identity(n_sources, dtype=bool, format="csr")
    steps = [
        (
            sp.kron(identity, query_fa.matrices[symbol].T, format="csr"),
            graph_fa.matrices[symbol],
        )
        for symbol in query_fa.matrices.keys() & graph_fa.matrices.keys()
    ]

    while frontier.nnz:
        reached = sp.csr_matrix(shape, dtype=bool)
        for left, right in steps:
            reached = reached + left @ frontier @ right
        reached = reached.astype(bool)
        frontier = (reached > visited).astype(bool)
        visited = (visited + frontier).astype(bool)

    result: set[tuple[int, int]] = set()
    rows, cols = visited.nonzero()
    for row, col in zip(rows, cols):
        source_idx, state = divmod(int(row), n_query)
        vertex = int(col)
        if state in query_fa.final_states and vertex in graph_fa.final_states:
            result.add(
                (
                    graph_fa.states[sources[source_idx]].value,
                    graph_fa.states[vertex].value,
                )
            )
    return result
