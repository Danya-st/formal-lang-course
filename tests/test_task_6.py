import networkx as nx
import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable
from project.task6 import hellings_based_cfpq, cfg_to_weak_normal_form


def _is_weak_cnf(cfg) -> bool:
    for production in cfg.productions:
        body = production.body
        if not body:
            continue
        if len(body) == 1 and isinstance(body[0], Terminal):
            continue
        if len(body) == 2 and all(isinstance(s, Variable) for s in body):
            continue
        return False
    return True


@pytest.mark.parametrize(
    "grammar, accepted_words, rejected_words",
    [
        (
            "S -> A B\nA -> C\nC -> epsilon | a\nB -> b | epsilon",
            [(), ("a",), ("b",), ("a", "b")],
            [("b", "a"), ("a", "a"), ("b", "b")],
        ),
        (
            "S -> a S b | epsilon",
            [(), ("a", "b"), ("a", "a", "b", "b")],
            [("a",), ("a", "b", "b"), ("b", "a")],
        ),
    ],
)
def test_conversion_preserves_language_and_weak_normal_form(
    grammar, accepted_words, rejected_words
):
    cfg = CFG.from_text(grammar)
    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.start_symbol == cfg.start_symbol
    assert _is_weak_cnf(normalized)
    for word in accepted_words:
        assert normalized.contains(list(word)), word
    for word in rejected_words:
        assert not normalized.contains(list(word)), word


def test_conversion_removes_unreachable_symbols():
    cfg = CFG.from_text("S -> a\nUnused -> b | epsilon")

    normalized = cfg_to_weak_normal_form(cfg)

    assert normalized.variables == {Variable("S")}
    assert normalized.terminals == {Terminal("a")}
    assert set(normalized.productions) == {Production(Variable("S"), [Terminal("a")])}


def test_conversion_of_empty_grammar():
    start = Variable("S")
    normalized = cfg_to_weak_normal_form(CFG(start_symbol=start))

    assert normalized.start_symbol == start
    assert not normalized.productions
    assert not normalized.contains([])


def test_hellings_nested_language_with_custom_start():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(2, 3, label="b")
    grammar = CFG.from_text("Q -> a Q b | a b", start_symbol=Variable("Q"))

    assert hellings_based_cfpq(grammar, graph) == {(0, 2)}
    assert hellings_based_cfpq(grammar, graph, {0}, {3}) == set()


def test_hellings_epsilon_gives_loops_on_every_vertex():
    graph = nx.DiGraph()
    graph.add_nodes_from([5, 6])

    assert hellings_based_cfpq(CFG.from_text("S -> epsilon"), graph) == {
        (5, 5),
        (6, 6),
    }


def test_hellings_matches_regular_chain():
    graph = nx.MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    graph.add_edge(20, 30, label="a")
    graph.add_edge(30, 40, label="a")

    result = hellings_based_cfpq(CFG.from_text("S -> a S | a"), graph)

    assert result == {(10, 20), (10, 30), (10, 40), (20, 30), (20, 40), (30, 40)}


def test_hellings_empty_graph_and_empty_language():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, label="a")

    assert hellings_based_cfpq(CFG.from_text("S -> a"), nx.DiGraph()) == set()
    assert hellings_based_cfpq(CFG(start_symbol=Variable("S")), graph) == set()
