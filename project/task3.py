from __future__ import annotations

from collections.abc import Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy import sparse

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    """represents a finite automaton, depicted as a sparse matrix of correspondences."""

    def __init__(
        self, automaton: NondeterministicFiniteAutomaton | None = None
    ) -> None:
        self.states: list[State] = []
        self.size = 0
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.matrices: dict[Symbol, sparse.csr_matrix] = {}
        self._index: dict[State, int] = {}
        if automaton is None:
            return

        self.states = list(
            dict.fromkeys(
                [*automaton.states, *automaton.start_states, *automaton.final_states]
            )
        )
        self.size = len(self.states)
        self._index = {state: i for i, state in enumerate(self.states)}
        self.start_states = {self._index[s] for s in automaton.start_states}
        self.final_states = {self._index[s] for s in automaton.final_states}
        self._load_transitions(automaton.to_dict())

    def _load_transitions(self, transitions: dict) -> None:
        buckets: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, by_symbol in transitions.items():
            for symbol, targets in by_symbol.items():
                if isinstance(targets, State):
                    targets = [targets]
                rows, cols = buckets.setdefault(symbol, ([], []))
                rows.extend([self._index[source]] * len(targets))
                cols.extend(self._index[target] for target in targets)
        self.matrices = {
            symbol: sparse.csr_matrix(
                ([True] * len(rows), (rows, cols)),
                shape=(self.size, self.size),
                dtype=bool,
            )
            for symbol, (rows, cols) in buckets.items()
        }

    def _row_vector(self, indices: Iterable[int]) -> sparse.csr_matrix:
        cols = list(indices)
        return sparse.csr_matrix(
            ([True] * len(cols), ([0] * len(cols), cols)),
            shape=(1, self.size),
            dtype=bool,
        )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """determines whether the automaton accepts a given string"""
        if not self.start_states:
            return False
        frontier = self._row_vector(self.start_states)
        for symbol in word:
            matrix = self.matrices.get(
                symbol if isinstance(symbol, Symbol) else Symbol(symbol)
            )
            if matrix is None:
                return False
            frontier = (frontier @ matrix).astype(bool)
            if frontier.nnz == 0:
                return False
        return any(frontier[0, final] for final in self.final_states)

    def reachability(self) -> sparse.csr_matrix:
        """Reflexive-transitive closure computed by squaring."""
        reach = sparse.identity(self.size, dtype=bool, format="csr")
        for matrix in self.matrices.values():
            reach = reach + matrix
        reach = reach.astype(bool)
        while True:
            doubled = (reach @ reach).astype(bool)
            if doubled.nnz == reach.nnz:
                return reach
            reach = doubled

    def is_empty(self) -> bool:
        """determines whether the language defined by an automaton is empty."""
        if not self.start_states or not self.final_states:
            return True
        reach = self.reachability()
        block = reach[sorted(self.start_states), :][:, sorted(self.final_states)]
        return block.nnz == 0


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    """intersection of two finite automata via the tensor product"""
    result = AdjacencyMatrixFA()
    result.states = [
        State((left, right))
        for left in automaton1.states
        for right in automaton2.states
    ]
    result.size = len(result.states)

    width = automaton2.size
    result.start_states = {
        i * width + j for i in automaton1.start_states for j in automaton2.start_states
    }
    result.final_states = {
        i * width + j for i in automaton1.final_states for j in automaton2.final_states
    }
    shared = automaton1.matrices.keys() & automaton2.matrices.keys()
    result.matrices = {
        symbol: sparse.kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        ).astype(bool)
        for symbol in shared
    }
    return result


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """performing regular queries on graphs: given a graph with specified start and end vertices
    and a regular expression, return those pairs of vertices (from the given start and end sets)
    that are connected by a path forming a word belonging to the language defined by the regular expression."""

    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    product = intersect_automata(graph_fa, regex_fa)
    reach = product.reachability()

    width = regex_fa.size
    pairs: set[tuple[int, int]] = set()
    for start in product.start_states:
        for final in product.final_states:
            if reach[start, final]:
                pairs.add(
                    (
                        graph_fa.states[start // width].value,
                        graph_fa.states[final // width].value,
                    )
                )
    return pairs
