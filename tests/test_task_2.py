import networkx as nx
import pytest
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

from project.task2 import graph_to_nfa, regex_to_dfa


def test_regex_to_dfa_returns_deterministic_dfa():
    dfa = regex_to_dfa("a b*")
    assert isinstance(dfa, DeterministicFiniteAutomaton)
    assert dfa.is_deterministic()


@pytest.mark.parametrize(
    "regex,accepted,rejected",
    [
        ("a b*", ["a", "ab", "abb"], ["", "b", "ba"]),
        ("a*", ["", "a", "aaa"], ["b", "ab"]),
        ("a|b", ["a", "b"], ["", "ab", "c"]),
        ("(a b)*", ["", "ab", "abab"], ["a", "aba"]),
    ],
)
def test_regex_to_dfa_accepts_and_rejects(regex, accepted, rejected):
    dfa = regex_to_dfa(regex)
    for word in accepted:
        assert dfa.accepts(list(word)), f"{regex!r} должен принимать {word!r}"
    for word in rejected:
        assert not dfa.accepts(list(word)), f"{regex!r} не должен принимать {word!r}"


def test_regex_to_dfa_result_is_minimal():
    dfa = regex_to_dfa("a b*")
    assert len(dfa.states) == len(dfa.minimize().states)


def test_regex_to_dfa_equivalent_expressions():
    first = regex_to_dfa("(a | b)*")
    second = regex_to_dfa("(a | b)* (a | b)*")
    assert first.is_equivalent_to(second)


def _cycle_graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 0, label="c")
    return graph


def test_graph_to_nfa_explicit_states():
    nfa = graph_to_nfa(_cycle_graph(), start_states={0}, final_states={2})
    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b", "c"])


def test_graph_to_nfa_defaults_to_all_nodes():
    nfa = graph_to_nfa(_cycle_graph())
    assert nfa.accepts([])
    assert nfa.accepts(["a"])
    assert nfa.accepts(["a", "b", "c"])


def test_graph_to_nfa_handles_nondeterminism():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 2, label="a")
    graph.add_edge(1, 3, label="b")
    graph.add_edge(2, 3, label="c")
    nfa = graph_to_nfa(graph, {0}, {3})
    assert nfa.accepts(["a", "b"])
    assert nfa.accepts(["a", "c"])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    nfa = graph_to_nfa(graph, {0}, {1})
    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])
    assert not nfa.accepts(["a", "a"])


def test_graph_to_nfa_does_not_modify_graph():
    graph = _cycle_graph()
    edges_before = graph.number_of_edges()
    graph_to_nfa(graph, {0}, {2})
    assert graph.number_of_edges() == edges_before


def test_graph_to_nfa_isolated_node_accepts_empty_word():
    graph = nx.MultiDiGraph()
    graph.add_node(0)
    nfa = graph_to_nfa(graph, {0}, {0})
    assert nfa.accepts([])
    assert not nfa.accepts(["a"])
