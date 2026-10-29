from collections import deque

import networkx as nx
import pytest
from pyformlang.finite_automaton import Symbol

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import (
    AdjacencyMatrixFA,
    intersect_automata,
    tensor_based_rpq,
)


def _words(alphabet, max_length):
    from itertools import product

    for length in range(max_length + 1):
        yield from (list(w) for w in product(alphabet, repeat=length))


@pytest.mark.parametrize("regex", ["a b*", "(a | b)* a", "epsilon", "a* b*"])
def test_accepts_agrees_with_pyformlang_on_short_words(regex):
    dfa = regex_to_dfa(regex)
    fa = AdjacencyMatrixFA(dfa)
    for word in _words(["a", "b", "c"], 4):
        assert fa.accepts(word) == dfa.accepts(word), (regex, word)


def test_accepts_rejects_unknown_symbol_in_middle_of_word():
    fa = AdjacencyMatrixFA(regex_to_dfa("a* b a*"))
    assert not fa.accepts(["a", "z", "b"])


def test_accepts_accepts_symbol_objects_and_raw_values():
    fa = AdjacencyMatrixFA(regex_to_dfa("a b"))
    assert fa.accepts([Symbol("a"), Symbol("b")])
    assert fa.accepts(["a", "b"])


def test_accepts_on_graph_with_parallel_labels():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="x")
    graph.add_edge(0, 1, label="y")
    graph.add_edge(1, 0, label="x")
    fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {0}))
    assert fa.accepts(["x", "x"])
    assert fa.accepts(["y", "x"])
    assert not fa.accepts(["y", "y"])


def test_is_empty_for_regex_without_final_reachability():
    fa = AdjacencyMatrixFA(regex_to_dfa("a"))
    fa.final_states = set()
    assert fa.is_empty()


def test_is_empty_with_long_chain_needs_closure():
    graph = nx.MultiDiGraph()
    for i in range(20):
        graph.add_edge(i, i + 1, label="a")
    assert not AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {20})).is_empty()
    assert AdjacencyMatrixFA(graph_to_nfa(graph, {20}, {0})).is_empty()


def test_intersection_matches_pyformlang_on_short_words():
    dfa1 = regex_to_dfa("(a | b)* a")
    dfa2 = regex_to_dfa("a (a | b)*")
    reference = dfa1.get_intersection(dfa2)
    result = intersect_automata(AdjacencyMatrixFA(dfa1), AdjacencyMatrixFA(dfa2))
    for word in _words(["a", "b"], 4):
        assert result.accepts(word) == reference.accepts(word), word


def test_intersection_is_commutative_on_words():
    left = AdjacencyMatrixFA(regex_to_dfa("a* b"))
    right = AdjacencyMatrixFA(regex_to_dfa("a b*"))
    ab = intersect_automata(left, right)
    ba = intersect_automata(right, left)
    for word in _words(["a", "b"], 4):
        assert ab.accepts(word) == ba.accepts(word), word


def test_intersection_with_empty_language_is_empty():
    full = AdjacencyMatrixFA(regex_to_dfa("(a | b)*"))
    nothing = AdjacencyMatrixFA(regex_to_dfa("a"))
    nothing.final_states = set()
    assert intersect_automata(full, nothing).is_empty()


def test_rpq_star_on_cycle_with_nonconsecutive_vertex_ids():
    graph = nx.MultiDiGraph()
    graph.add_edge(7, 42, label="a")
    graph.add_edge(42, 7, label="a")
    result = tensor_based_rpq("a*", graph, {7}, {7, 42})
    assert result == {(7, 7), (7, 42)}


def test_rpq_respects_start_and_final_filters():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 2, label="a")
    assert tensor_based_rpq("a", graph, {0}, {1}) == {(0, 1)}
    assert tensor_based_rpq("a", graph, {1}, {2}) == set()


def _reference_rpq(regex, graph, starts, finals):
    """Независимый обход по парам (вершина, состояние DFA)."""
    dfa = regex_to_dfa(regex)
    result = set()
    for source in starts:
        seen = {(source, s) for s in dfa.start_states}
        queue = deque(seen)
        while queue:
            vertex, state = queue.popleft()
            if vertex in finals and state in dfa.final_states:
                result.add((source, vertex))
            for _, target, data in graph.out_edges(vertex, data=True):
                for nxt in dfa(state, data["label"]):
                    if (target, nxt) not in seen:
                        seen.add((target, nxt))
                        queue.append((target, nxt))
    return result


@pytest.mark.parametrize("regex", ["a b", "(a | b)*", "a* b a*", "(a b)*"])
def test_rpq_matches_reference_search(regex):
    graph = nx.MultiDiGraph()
    edges = [
        (0, 1, "a"),
        (1, 2, "b"),
        (2, 0, "a"),
        (2, 3, "b"),
        (3, 3, "a"),
        (1, 1, "b"),
    ]
    for u, v, label in edges:
        graph.add_edge(u, v, label=label)
    nodes = {0, 1, 2, 3}
    assert tensor_based_rpq(regex, graph, nodes, nodes) == _reference_rpq(
        regex, graph, nodes, nodes
    )


def test_rpq_empty_graph_returns_empty_set():
    assert tensor_based_rpq("a*", nx.MultiDiGraph(), set(), set()) == set()
