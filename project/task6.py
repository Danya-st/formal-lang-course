from pyformlang.cfg import CFG, Production, Terminal
from collections import defaultdict, deque
import networkx as nx

__all__ = ["cfg_to_weak_normal_form"]


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """transform a context-free grammar into a weak Chomsky normal form"""
    normal_form = cfg.to_normal_form()
    productions = set(normal_form.productions)
    productions.update(Production(v, []) for v in cfg.get_nullable_symbols())
    weak = CFG(start_symbol=cfg.start_symbol, productions=productions)
    return weak.remove_useless_symbols()


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """solves the reachability problem between all pairs of vertices for a given graph and a given context-free grammar"""
    nodes = set(graph.nodes)
    starts = start_nodes if start_nodes else nodes
    finals = final_nodes if final_nodes else nodes

    normal = cfg_to_weak_normal_form(cfg)

    terminal_rules: dict[str, set[str]] = defaultdict(set)
    binary_rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    empty_rules: set[str] = set()

    for production in normal.productions:
        body = production.body
        head = production.head.value
        if not body:
            empty_rules.add(head)
        elif len(body) == 1 and isinstance(body[0], Terminal):
            terminal_rules[body[0].value].add(head)
        elif len(body) == 2:
            binary_rules[(body[0].value, body[1].value)].add(head)

    relation: set[tuple[int, str, int]] = set()
    for u, v, data in graph.edges(data=True):
        for head in terminal_rules.get(data.get("label"), ()):
            relation.add((u, head, v))
    for v in nodes:
        for head in empty_rules:
            relation.add((v, head, v))

    by_source: dict[int, set[tuple[str, int]]] = defaultdict(set)
    by_target: dict[int, set[tuple[str, int]]] = defaultdict(set)
    for u, a, v in relation:
        by_source[u].add((a, v))
        by_target[v].add((a, u))

    changed = True
    while changed:
        changed = False
        new = set()
        for u, a, v in relation:
            for b, w in by_source[v]:
                for head in binary_rules.get((a, b), ()):
                    if (u, head, w) not in relation:
                        new.add((u, head, w))
        if new:
            for u, head, w in new:
                if (u, head, w) not in relation:
                    relation.add((u, head, w))
                    by_source[u].add((head, w))
                    by_target[w].add((head, u))
            changed = True

    start_symbol = normal.start_symbol.value
    return {
        (u, v)
        for u, head, v in relation
        if head == start_symbol and u in starts and v in finals
    }
