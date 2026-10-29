from typing import Set
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """constructs a minimal DFA for a given regular expression"""
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: Set[int] | None = None,
    final_states: Set[int] | None = None,
) -> NondeterministicFiniteAutomaton:
    """constructs a nondeterministic finite automaton from a graph"""
    nfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    starts = start_states if start_states else set(graph.nodes)
    finals = final_states if final_states else set(graph.nodes)

    for state in starts:
        nfa.add_start_state(state)
    for state in finals:
        nfa.add_final_state(state)

    return nfa
