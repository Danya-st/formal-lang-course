import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable

from project.task6 import hellings_based_cfpq
from project.task7 import matrix_based_cfpq


def test_matrix_cfpq_on_digraph_with_nonconsecutive_vertices():
    graph = nx.DiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(40, 50, label="b")
    graph.add_edge(50, 60, label="b")
    cfg = CFG.from_text("S -> a S b | a b")

    assert matrix_based_cfpq(cfg, graph) == {(20, 40), (10, 50)}


@pytest.mark.parametrize("nodes", [None, set()])
def test_empty_boundaries_select_all_vertices(nodes):
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")

    assert matrix_based_cfpq(CFG.from_text("S -> a"), graph, nodes, nodes) == {
        (10, 20),
        (20, 30),
    }


@pytest.mark.parametrize("starts, finals", [({99}, None), (None, {99})])
def test_unknown_boundary_vertices_raise(starts, finals):
    graph = nx.DiGraph()
    graph.add_node(10)
    with pytest.raises(ValueError, match="graph vertices"):
        matrix_based_cfpq(CFG.from_text("S -> $"), graph, starts, finals)


def test_epsilon_loops_on_isolated_vertices_and_filters():
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    result = matrix_based_cfpq(CFG.from_text("S -> $"), graph, {0, 2}, {1, 2})
    assert result == {(2, 2)}


def test_multicharacter_labels_and_custom_start_symbol():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 2, label="right")
    cfg = CFG.from_text(
        "Query -> left right\nS -> ignored", start_symbol=Variable("Query")
    )

    assert matrix_based_cfpq(cfg, graph) == {(0, 2)}


@pytest.mark.parametrize("text", ["S -> a | $", "S -> a"])
def test_empty_graph(text):
    assert matrix_based_cfpq(CFG.from_text(text), nx.MultiDiGraph()) == set()


def test_empty_grammar_gives_empty_result():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    assert matrix_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()


def test_many_derivations_of_same_pair_are_not_lost():
    graph = nx.MultiDiGraph()
    for middle in range(1, 257):
        graph.add_edge(0, middle, label="a")
        graph.add_edge(middle, 257, label="b")

    assert matrix_based_cfpq(CFG.from_text("S -> a b"), graph) == {(0, 257)}


def test_self_loop_grammar_reaches_fixed_point():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 0, label="a")
    cfg = CFG.from_text("S -> S S | a | $")

    assert matrix_based_cfpq(cfg, graph) == hellings_based_cfpq(cfg, graph)


@pytest.mark.parametrize(
    "text",
    ["S -> a S b | S S | $", "S -> A B\nA -> a A | $\nB -> b B | $"],
)
def test_matches_hellings_with_boundaries(text):
    graph = nx.MultiDiGraph()
    edges = [(0, 1, "a"), (1, 2, "a"), (2, 3, "b"), (3, 0, "b"), (2, 2, "b")]
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    cfg = CFG.from_text(text)
    starts, finals = {0, 1, 2}, {2, 3}

    assert matrix_based_cfpq(cfg, graph, starts, finals) == hellings_based_cfpq(
        cfg, graph, starts, finals
    )
