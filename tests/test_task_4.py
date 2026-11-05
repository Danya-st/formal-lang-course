import random
from collections import deque

import networkx as nx
import pytest

from project.task2 import regex_to_dfa
from project.task3 import tensor_based_rpq
from project.task4 import ms_bfs_based_rpq


def _reference_rpq(regex, graph, starts, finals):
    """Независимый обход в ширину по парам (вершина, состояние DFA)."""
    dfa = regex_to_dfa(regex)
    transitions = dfa.to_dict()
    result = set()
    for source in starts:
        seen = {(source, dfa.start_state)}
        queue = deque(seen)
        while queue:
            vertex, state = queue.popleft()
            if vertex in finals and state in dfa.final_states:
                result.add((source, vertex))
            for _, target, data in graph.out_edges(vertex, data=True):
                for symbol, next_state in transitions.get(state, {}).items():
                    if (
                        str(symbol) == data["label"]
                        and (target, next_state) not in seen
                    ):
                        seen.add((target, next_state))
                        queue.append((target, next_state))
    return result


def test_filters_by_start_final_and_regex():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="b")

    assert ms_bfs_based_rpq("a b*", graph, {0, 1}, {1, 2, 3}) == {
        (0, 1),
        (0, 2),
        (0, 3),
    }


def test_cycles_and_empty_word():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")
    graph.add_node(2)

    assert ms_bfs_based_rpq("a*", graph, {0, 2}, {0, 1, 2}) == {
        (0, 0),
        (0, 1),
        (2, 2),
    }


def test_sources_both_reach_shared_target():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(10, 1, label="a")

    assert ms_bfs_based_rpq("a b", graph, {0, 10}, {2}) == {(0, 2), (10, 2)}


def test_long_path_with_several_sources():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="open")
    graph.add_edge(1, 2, label="read")
    graph.add_edge(2, 1, label="retry")
    graph.add_edge(2, 3, label="close")
    graph.add_edge(10, 11, label="open")
    graph.add_edge(11, 3, label="close")

    result = ms_bfs_based_rpq("open read (retry read)* close", graph, {0, 10}, {3})

    assert result == {(0, 3)}


def test_zero_length_paths_only_for_allowed_pairs():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="a")

    assert ms_bfs_based_rpq("(a b)*", graph, {0, 1, 2}, {0, 2}) == {
        (0, 0),
        (0, 2),
        (2, 2),
    }


def test_empty_boundary_sets_mean_all_nodes():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="c")

    assert ms_bfs_based_rpq("a|b|c", graph, set(), set()) == {
        (0, 1),
        (1, 2),
        (2, 0),
    }


def test_symbol_absent_from_graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    assert ms_bfs_based_rpq("x", graph, {0}, {1}) == set()


def test_empty_graph():
    assert ms_bfs_based_rpq("a", nx.MultiDiGraph(), {1}, {2}) == set()


def test_does_not_modify_inputs():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="b")
    starts, finals = {0}, {1}
    before = graph.copy()

    ms_bfs_based_rpq("(a b)* a", graph, starts, finals)

    assert nx.utils.graphs_equal(graph, before)
    assert starts == {0}
    assert finals == {1}


@pytest.mark.parametrize("seed", range(10))
def test_matches_reference_and_tensor_on_random_graphs(seed):
    rng = random.Random(seed)
    labels = ["a", "b", "c"]
    size = rng.randint(1, 6)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(size))
    for _ in range(rng.randint(0, 12)):
        graph.add_edge(
            rng.randrange(size), rng.randrange(size), label=rng.choice(labels)
        )
    starts = {n for n in range(size) if rng.random() < 0.5} or {0}
    finals = {n for n in range(size) if rng.random() < 0.5} or {0}

    for regex in ["a b*", "(a|b)* c", "a* b* c*", "(a b)*", "c a*"]:
        expected = _reference_rpq(regex, graph, starts, finals)
        assert ms_bfs_based_rpq(regex, graph, starts, finals) == expected
        assert tensor_based_rpq(regex, graph, starts, finals) == expected
