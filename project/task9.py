from collections import defaultdict

import networkx as nx
from pyformlang.rsa import RecursiveAutomaton


def val(x):
    return getattr(x, "value", x)


def compile_box(dfa) -> tuple:
    transitions = defaultdict(list)
    for src, out in dfa.to_dict().items():
        for sym, dst in out.items():
            targets = dst if isinstance(dst, (set, frozenset, list)) else [dst]
            for t in targets:
                transitions[src].append((val(sym), t))
    return dfa.start_state, set(dfa.final_states), transitions


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """a function based on the Generalized LL algorithm (which operates using RSMs)
    that determines the reachability problem between all pairs of vertices
    for a given graph and a given context-free grammar.
    """
    nodes = set(graph.nodes)
    starts = nodes if start_nodes is None else set(start_nodes) & nodes
    finals = nodes if final_nodes is None else set(final_nodes) & nodes
    if not starts or not finals:
        return set()

    boxes = {val(label): compile_box(box.dfa) for label, box in rsm.boxes.items()}
    init = val(rsm.initial_label)
    if init not in boxes:
        return set()

    adj = defaultdict(lambda: defaultdict(list))  # vertex -> label -> [targets]
    for u, v, label in graph.edges(data="label"):
        if label is not None:
            adj[u][label].append(v)

    gss_edges = defaultdict(set)  # callee -> {(return_state, caller)}
    pops = defaultdict(set)  # GSS node -> {vertex}
    seen = set()
    stack = []

    def push(node, state, vertex):
        d = (node, state, vertex)
        if d not in seen:
            seen.add(d)
            stack.append(d)

    for s in starts:
        push((init, s), boxes[init][0], s)

    while stack:
        node, state, vertex = stack.pop()
        label = node[0]
        _, box_finals, trans = boxes[label]

        # Pop: the box is finished at `vertex`; return to all known callers.
        if state in box_finals and vertex not in pops[node]:
            pops[node].add(vertex)
            for ret_state, caller in gss_edges[node]:
                push(caller, ret_state, vertex)

        for sym, nxt in trans.get(state, ()):
            if sym in boxes:  # nonterminal -> call
                callee = (sym, vertex)
                edge = (nxt, node)
                if edge not in gss_edges[callee]:
                    gss_edges[callee].add(edge)
                    # callee may already have finished: replay its results
                    for w in pops[callee]:
                        push(node, nxt, w)
                push(callee, boxes[sym][0], vertex)
            else:  # terminal -> consume a graph edge
                for w in adj[vertex].get(sym, ()):
                    push(node, nxt, w)

    return {(s, v) for s in starts for v in pops[(init, s)] if v in finals}
