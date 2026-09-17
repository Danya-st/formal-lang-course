import cfpq_data as cd
import networkx as nx
import pydot
import pathlib
from dataclasses import dataclass

@dataclass
class GraphInfo:
    """Contains graph statistics"""
    number_of_nodes:int
    number_of_edges:int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    """gives basic information about graph"""

    graph = cd.graph_from_csv(cd.download(graph_name))

    labels = {label for _, _, label in graph.edges(data="label") if label is not None}

    return GraphInfo(number_of_nodes = graph.number_of_nodes(), number_of_edges = graph.number_of_edges(), labels = labels)


def create_and_save_two_cycles_graph(n:int,m:int,labels:tuple[str,str],path:str | pathlib.Path) -> nx.MultiDiGraph:
    """constructs a two cycles graph and saves it in DOT format"""
    graph = cd.labeled_two_cycles_graph(n, m, labels=labels)

    dot = pydot.Dot(graph_type = "digraph")

    for node in graph.nodes:
        dot.add_node(pydot.Node(str(node)))

    for source, target, label in graph.edges(data="label"):
        dot.add_edge(pydot.Edge(str(source), str(target), label=label))

    dot.write_raw(str(path))

    return graph
