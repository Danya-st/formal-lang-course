import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.task3 import AdjacencyMatrixFA
from project.task2 import graph_to_nfa


def _sym_key(symbol) -> str:
    return str(getattr(symbol, "value", symbol))


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """CFG -> RSM via a text-based representation in pyformlang."""
    if not cfg.productions:
        raise ValueError("Cannot convert a grammar without productions to RSM")
    return RecursiveAutomaton.from_text(
        cfg.to_text(), start_symbol=Symbol(cfg.start_symbol.value)
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """EBNF -> RSM."""
    return RecursiveAutomaton.from_text(ebnf)


def _transitive_closure(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Reflexive-transitive closure of a Boolean matrix via squaring."""
    n = adj.shape[0]
    reach = ((adj + sp.identity(n, dtype=bool)) > 0).tocsr()
    while True:
        new = (reach @ reach) > 0
        if new.nnz == reach.nnz:
            return reach
        reach = new.tocsr()


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """a function based on a tensor algorithm that solves the all-pairs reachability problem for a given graph and a given context-free grammar."""
    nodes = set(graph.nodes)
    start_nodes = start_nodes or nodes
    final_nodes = final_nodes or nodes

    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    index = graph_automaton._index
    dim = graph_automaton.size

    graph_matrices = {
        _sym_key(label): matrix for label, matrix in graph_automaton.matrices.items()
    }
    for label in rsm.labels:
        graph_matrices.setdefault(
            _sym_key(label), sp.csr_matrix((dim, dim), dtype=bool)
        )

    boxes = [(_sym_key(l), AdjacencyMatrixFA(b.dfa)) for l, b in rsm.boxes.items()]
    symbols = {_sym_key(s) for _, a in boxes for s in a.matrices}

    rsm_matrices = {}
    for sym in symbols:
        blocks = [
            a.matrices.get(sym, sp.csr_matrix((a.size, a.size), dtype=bool))
            for _, a in boxes
        ]
        rsm_matrices[sym] = sp.block_diag(blocks, format="csr")

    box_info, offset = {}, 0
    for label, a in boxes:
        starts = [offset + s for s in a.start_states]
        finals = [offset + f for f in a.final_states]
        box_info[label] = (starts, finals)
        offset += a.size
    states_count = offset

    initial_label = _sym_key(rsm.initial_label)
    graph_matrices.setdefault(initial_label, sp.csr_matrix((dim, dim), dtype=bool))

    if states_count == 0 or dim == 0:
        return set()

    while True:
        product = sp.csr_matrix((dim * states_count, dim * states_count), dtype=bool)
        for sym, rsm_m in rsm_matrices.items():
            gm = graph_matrices.get(sym)
            if gm is None or gm.nnz == 0:
                continue
            product = product + sp.kron(gm, rsm_m, format="csr")
        product = (product > 0).tocsr()

        reach = _transitive_closure(product)

        changed = False
        for label, (starts, finals) in box_info.items():
            to_add = sp.csr_matrix((dim, dim), dtype=bool)
            for st in starts:
                for f in finals:
                    to_add = to_add + reach[st::states_count, f::states_count]
            to_add = (to_add > 0).tocsr()

            added = (to_add > graph_matrices[label]).tocsr()
            if added.nnz:
                graph_matrices[label] = ((graph_matrices[label] + added) > 0).tocsr()
                changed = True

        if not changed:
            break

    result = graph_matrices[initial_label]
    return {
        (s, t)
        for s in start_nodes
        for t in final_nodes
        if s in index and t in index and result[index[s], index[t]]
    }
