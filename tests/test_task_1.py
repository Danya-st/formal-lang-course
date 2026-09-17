import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import networkx as nx
import pydot
import pytest

from project.task_1 import GraphInfo, create_and_save_two_cycles_graph, get_graph_info


def test_returns_graph_info_instance():
    info = get_graph_info("bzip")
    assert isinstance(info, GraphInfo)


def test_fields_have_correct_types():
    info = get_graph_info("bzip")
    assert isinstance(info.number_of_nodes, int)
    assert isinstance(info.number_of_edges, int)
    assert isinstance(info.labels, set)
    assert all(isinstance(label, str) for label in info.labels)


def test_bzip_graph():
    info = get_graph_info("bzip")
    assert info.number_of_nodes == 632
    assert info.number_of_edges == 556
    assert info.labels == {"a", "d"}


def test_generations_graph():
    info = get_graph_info("generations")
    assert info.number_of_nodes == 129
    assert info.number_of_edges == 273
    assert len(info.labels) == 17
    assert "hasChild" in info.labels
    assert "hasParent" in info.labels


def test_unknown_graph_raises_error():
    with pytest.raises(FileNotFoundError):
        get_graph_info("nonexistent_graph_xyz")


def test_returns_multidigraph(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    graph = create_and_save_two_cycles_graph(3, 2, ("a", "b"), out)
    assert isinstance(graph, nx.MultiDiGraph)


def test_creates_dot_file(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    create_and_save_two_cycles_graph(3, 4, ("x", "y"), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_node_count(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    graph = create_and_save_two_cycles_graph(3, 2, ("a", "b"), out)
    assert graph.number_of_nodes() == 6  # 3 + 2 + 1


def test_edge_count(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    graph = create_and_save_two_cycles_graph(3, 2, ("a", "b"), out)
    assert graph.number_of_edges() == 7  # 3 + 2 + 2


def test_labels_in_dot_file(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    create_and_save_two_cycles_graph(3, 4, ("first", "second"), out)

    (dot_graph,) = pydot.graph_from_dot_file(str(out))
    edge_labels = {edge.get_label().strip('"') for edge in dot_graph.get_edges()}
    assert edge_labels == {"first", "second"}


def test_round_trip(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    original = create_and_save_two_cycles_graph(3, 2, ("x", "y"), out)

    (dot_graph,) = pydot.graph_from_dot_file(str(out))
    assert len(dot_graph.get_nodes()) == original.number_of_nodes()
    assert len(dot_graph.get_edges()) == original.number_of_edges()


def test_dot_graph_type(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    create_and_save_two_cycles_graph(3, 2, ("a", "b"), out)
    (dot_graph,) = pydot.graph_from_dot_file(str(out))
    assert dot_graph.get_type() == "digraph"


def test_accepts_pathlib_path(tmp_path: pathlib.Path):
    out = tmp_path / "g.dot"
    graph = create_and_save_two_cycles_graph(2, 3, ("a", "b"), out)
    assert out.exists()
    assert isinstance(graph, nx.MultiDiGraph)


def test_accepts_str_path(tmp_path: pathlib.Path):
    out = str(tmp_path / "g.dot")
    graph = create_and_save_two_cycles_graph(2, 3, ("a", "b"), out)
    assert pathlib.Path(out).exists()
    assert isinstance(graph, nx.MultiDiGraph)
