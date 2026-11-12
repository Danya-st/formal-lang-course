import networkx as nx
from collections import defaultdict
from pyformlang.cfg import CFG
from scipy.sparse import csr_matrix, eye

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """based on a matrix algorithm, which solves the all-pairs reachability problem for a given graph and a given context-free grammar"""
    grammar = cfg_to_weak_normal_form(cfg)

    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    vertices = set(nodes)
    starts = set(start_nodes) if start_nodes else vertices
    finals = set(final_nodes) if final_nodes else vertices
    if not (starts | finals) <= vertices:
        raise ValueError("Start and final states must be graph vertices")

    size = len(nodes)
    matrices = {
        variable: csr_matrix((size, size), dtype=bool)
        for variable in grammar.variables | {grammar.start_symbol}
    }

    terminal_rules = defaultdict(list)
    epsilon_heads = []
    binary_rules = []
    for production in grammar.productions:
        body = production.body
        if not body:
            epsilon_heads.append(production.head)
        elif len(body) == 1:
            terminal_rules[body[0].value].append(production.head)
        elif len(body) == 2:
            binary_rules.append((production.head, body[0], body[1]))
        else:
            raise ValueError("Grammar is not in weak Chomsky normal form")

    edge_rows = defaultdict(list)
    edge_cols = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        label = data.get("label")
        edge_rows[label].append(index[source])
        edge_cols[label].append(index[target])

    for label, rows in edge_rows.items():
        heads = terminal_rules.get(label)
        if not heads:
            continue
        edge_matrix = csr_matrix(
            ([True] * len(rows), (rows, edge_cols[label])),
            shape=(size, size),
            dtype=bool,
        )
        for head in heads:
            matrices[head] = matrices[head].maximum(edge_matrix)

    if size and epsilon_heads:
        identity = eye(size, format="csr", dtype=bool)
        for head in epsilon_heads:
            matrices[head] = matrices[head].maximum(identity)

    binary_by_body = defaultdict(list)
    for head, left, right in binary_rules:
        binary_by_body[(left, right)].append(head)

    changed = {variable for variable, matrix in matrices.items() if matrix.nnz}
    while changed:
        next_changed = set()
        for (left, right), heads in binary_by_body.items():
            if left not in changed and right not in changed:
                continue
            product = matrices[left] @ matrices[right]
            if product.nnz == 0:
                continue
            for head in heads:
                current = matrices[head]
                updated = current.maximum(product)
                if updated.nnz != current.nnz:
                    matrices[head] = updated
                    next_changed.add(head)
        changed = next_changed

    rows, cols = matrices[grammar.start_symbol].nonzero()
    return {
        (nodes[row], nodes[col])
        for row, col in zip(rows, cols)
        if nodes[row] in starts and nodes[col] in finals
    }
