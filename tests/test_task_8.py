"""Test cases for the tensor-based CFPQ solution."""

import networkx as nx
import pytest
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import Symbol

from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_cfg_conversion_preserves_start_symbol_and_productions() -> None:
    cfg = CFG.from_text(
        "Query -> left Query right | $\nUnused -> other",
        start_symbol=Variable("Query"),
    )
    rsm = cfg_to_rsm(cfg)
    box = rsm.get_box(Symbol("Query"))

    assert rsm.initial_label == Symbol("Query")
    assert box.dfa.accepts([])
    assert box.dfa.accepts(["left", "Query", "right"])
    assert not box.dfa.accepts(["left", "right"])


def test_ebnf_conversion_preserves_initial_label() -> None:
    rsm = ebnf_to_rsm("S -> (left right)*")
    assert rsm.initial_label == Symbol("S")


def test_tensor_cfpq_handles_recursion_nullable_calls_and_node_filters() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="b")
    graph.add_edge(40, 50, label="b")
    graph.add_node(60)
    rsm = cfg_to_rsm(CFG.from_text("S -> a S b | $"))

    assert tensor_based_cfpq(rsm, graph, {10, 20, 60}, {40, 50, 60}) == {
        (10, 50),
        (20, 40),
        (60, 60),
    }


def test_tensor_cfpq_handles_ebnf_and_parallel_edges() -> None:
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="ignored")
    graph.add_edge(0, 1, label="left")
    graph.add_edge(1, 2, label="right")
    graph.add_edge(2, 3, label="left")
    graph.add_edge(3, 4, label="right")
    rsm = ebnf_to_rsm("S -> (left right)*")

    assert tensor_based_cfpq(rsm, graph, {0}, {2, 4}) == {(0, 2), (0, 4)}


def test_tensor_cfpq_with_empty_graph() -> None:
    graph = nx.DiGraph()
    assert tensor_based_cfpq(cfg_to_rsm(CFG.from_text("S -> $")), graph) == set()
