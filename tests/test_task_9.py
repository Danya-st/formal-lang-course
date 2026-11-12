import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.task8 import cfg_to_rsm, ebnf_to_rsm
from project.task9 import gll_based_cfpq


def graph_of(edges, nodes=(), multi=True) -> nx.DiGraph:
    graph = nx.MultiDiGraph() if multi else nx.DiGraph()
    graph.add_nodes_from(nodes)
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def rsm_of(text: str, **kwargs):
    return cfg_to_rsm(CFG.from_text(text, **kwargs))


def test_nonconsecutive_vertices() -> None:
    graph = graph_of(
        [(10, 20, "a"), (20, 30, "a"), (30, 40, "b"), (40, 50, "b"), (50, 60, "b")],
        multi=False,
    )
    assert gll_based_cfpq(rsm_of("S -> a S b | a b"), graph) == {(20, 40), (10, 50)}


def test_nullable_start_symbol_and_node_filters() -> None:
    graph = graph_of(
        [(10, 20, "a"), (20, 30, "a"), (30, 40, "b"), (40, 50, "b")], nodes=[60]
    )
    result = gll_based_cfpq(rsm_of("S -> a S b | $"), graph, {10, 20, 60}, {40, 50, 60})
    assert result == {(10, 50), (20, 40), (60, 60)}


def test_epsilon_only_on_isolated_vertices_with_filters() -> None:
    graph = graph_of([], nodes=[0, 1, 2])
    assert gll_based_cfpq(rsm_of("S -> $"), graph, {0, 2}, {1, 2}) == {(2, 2)}


@pytest.mark.parametrize("text", ["S -> a | $", "S -> a"])
def test_empty_graph(text: str) -> None:
    assert gll_based_cfpq(rsm_of(text), nx.MultiDiGraph()) == set()


def test_custom_start_symbol() -> None:
    graph = graph_of([(0, 1, "ignored"), (0, 1, "left"), (1, 2, "right")])
    rsm = rsm_of("Query -> left right\nS -> ignored", start_symbol=Variable("Query"))
    assert gll_based_cfpq(rsm, graph) == {(0, 2)}


def test_terminal_with_nonterminal_name_is_not_a_call() -> None:
    graph = graph_of([(0, 1, "A"), (3, 1, "a"), (1, 2, "b")])
    assert gll_based_cfpq(rsm_of("S -> A b\nA -> a"), graph) == {(3, 2)}


@pytest.mark.parametrize(
    "starts, finals, expected",
    [
        (set(), None, set()),
        (None, set(), set()),
        ({0}, None, {(0, 1)}),
        (None, {1}, {(0, 1)}),
        (None, {0}, set()),
        ({0}, {1}, {(0, 1)}),
        ({1}, {1}, set()),
        ({42}, {99}, set()),
    ],
)
def test_node_set_arguments(starts, finals, expected) -> None:
    graph = graph_of([(0, 1, "a")])
    assert gll_based_cfpq(rsm_of("S -> a"), graph, starts, finals) == expected


def test_direct_left_recursion_on_cycle() -> None:
    graph = graph_of([(0, 1, "a"), (1, 0, "a")], multi=False)
    assert gll_based_cfpq(rsm_of("S -> S a | a"), graph) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_indirect_left_recursion() -> None:
    graph = graph_of([(0, 1, "a"), (1, 0, "b")], multi=False)
    assert gll_based_cfpq(rsm_of("S -> A b\nA -> S a | a"), graph) == {(0, 0)}


def test_mutual_recursion() -> None:
    graph = graph_of([(0, 1, "a"), (1, 2, "b"), (2, 3, "a"), (3, 4, "b")])
    result = gll_based_cfpq(rsm_of("S -> a B | a\nB -> b S | b"), graph)
    assert result == {(0, 1), (0, 2), (0, 3), (0, 4), (2, 3), (2, 4)}


def test_final_state_with_outgoing_transitions() -> None:
    graph = graph_of([(0, 1, "a"), (1, 2, "a"), (2, 3, "a")])
    assert gll_based_cfpq(rsm_of("S -> a | a S"), graph) == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }


def test_nullable_nonterminal_inside_rule() -> None:
    graph = graph_of([(0, 1, "a"), (1, 2, "b"), (0, 3, "b")])
    assert gll_based_cfpq(rsm_of("S -> A b\nA -> a | $"), graph) == {
        (0, 2),
        (1, 2),
        (0, 3),
    }


def test_hidden_left_recursion_through_nullable() -> None:
    graph = graph_of([(0, 1, "b"), (1, 2, "b")])
    result = gll_based_cfpq(rsm_of("S -> A S b | $\nA -> $"), graph)
    assert result == {(i, j) for i in range(3) for j in range(i, 3)}


def test_ambiguous_nullable_left_recursive_grammar() -> None:
    graph = graph_of([(0, 1, "a"), (1, 0, "a")], nodes=[2])
    result = gll_based_cfpq(rsm_of("S -> S S | a | $"), graph)
    assert result == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)}


def test_nested_nonterminals_share_calls() -> None:
    graph = graph_of([(i, i + 1, label) for i, label in enumerate("abab")])
    result = gll_based_cfpq(rsm_of("S -> A B S | A B\nA -> a\nB -> b"), graph)
    assert result == {(0, 2), (2, 4), (0, 4)}


def test_late_caller_receives_already_popped_results() -> None:
    chain = [10, 11, 12, 13, 14, 15, 16, 17, 18, 0]
    edges = [(0, 2, "x"), (2, 3, "a")] + [(u, v, "p") for u, v in zip(chain, chain[1:])]
    ps = " ".join(["p"] * 9)
    rsm = rsm_of(f"S -> T | {ps} T\nT -> x A\nA -> a")
    assert gll_based_cfpq(rsm, graph_of(edges), {0, 10}, {3}) == {(0, 3), (10, 3)}


def test_ebnf_star_and_parallel_edges() -> None:
    graph = graph_of(
        [
            (0, 1, "ignored"),
            (0, 1, "left"),
            (1, 2, "right"),
            (2, 3, "left"),
            (3, 4, "right"),
        ]
    )
    result = gll_based_cfpq(ebnf_to_rsm("S -> (left right)*"), graph, {0}, {2, 4})
    assert result == {(0, 2), (0, 4)}


@pytest.mark.parametrize(
    "text, edges, expected",
    [
        (
            "S -> a (b | c)*",
            [(0, 1, "a"), (1, 2, "b"), (2, 3, "c"), (3, 4, "d")],
            {(0, 1), (0, 2), (0, 3)},
        ),
        (
            "S -> (a (b | c))*",
            [(0, 1, "a"), (1, 2, "c"), (2, 3, "a"), (3, 4, "b")],
            {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 2), (2, 4), (0, 4)},
        ),
    ],
)
def test_ebnf_alternation_and_nested_groups(text, edges, expected) -> None:
    assert gll_based_cfpq(ebnf_to_rsm(text), graph_of(edges)) == expected


BOUNDARY_EDGES = [(0, 1, "a"), (1, 2, "a"), (2, 3, "b"), (3, 0, "b"), (2, 2, "b")]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("S -> a S b | S S | $", {(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)}),
        (
            "S -> A B\nA -> a A | $\nB -> b B | $",
            {(0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3)},
        ),
    ],
)
def test_start_and_final_boundaries(text, expected) -> None:
    graph = graph_of(BOUNDARY_EDGES)
    assert gll_based_cfpq(rsm_of(text), graph, {0, 1, 2}, {2, 3}) == expected


def test_many_parallel_paths_collapse_to_one_pair() -> None:
    edges = []
    for middle in range(1, 257):
        edges += [(0, middle, "a"), (middle, 257, "b")]
    assert gll_based_cfpq(rsm_of("S -> a b"), graph_of(edges)) == {(0, 257)}


def test_long_chain_no_recursion_error() -> None:
    n = 1000
    edges = [(i, i + 1, "a") for i in range(n)]
    edges += [(i, i + 1, "b") for i in range(n, 2 * n)]
    result = gll_based_cfpq(rsm_of("S -> a S b | a b"), graph_of(edges, multi=False))
    assert result == {(n - k, n + k) for k in range(1, n + 1)}


def test_result_type_repeatable_and_inputs_not_mutated() -> None:
    graph = graph_of([(0, 1, "a"), (1, 0, "a")])
    rsm = rsm_of("S -> S a | a")
    before = sorted(graph.edges(data=True))

    first = gll_based_cfpq(rsm, graph)
    second = gll_based_cfpq(rsm, graph)

    assert isinstance(first, set)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in first)
    assert first == second == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert sorted(graph.edges(data=True)) == before
