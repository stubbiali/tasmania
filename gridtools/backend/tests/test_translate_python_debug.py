import pytest
import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.backend.translate_to_python_debug as t2pdb


def test_generate_indexing_string_for_named_expression():
    assert t2pdb.generate_indexing_string_for_named_expression([0]) == "[i]"
    assert t2pdb.generate_indexing_string_for_named_expression([0, 0]) == "[i,j]"
    assert t2pdb.generate_indexing_string_for_named_expression([1]) == "[i+1]"
    assert t2pdb.generate_indexing_string_for_named_expression([0, -1]) == "[i,j-1]"
    assert t2pdb.generate_indexing_string_for_named_expression([2, 2]) == "[i+2,j+2]"
    assert t2pdb.generate_indexing_string_for_named_expression([0, 0, 1]) == "[i,j,k+1]"


def test_empty_graph():
    """
    Definitions: {} <-- empty graph
    """
    graph = nx.MultiDiGraph()
    stage_src = t2pdb.translate_graph(graph)
    assert stage_src == []


def test_no_inputs():
    """
    Definitions:
    B = 10
    """
    b = irg.NodeNamedExpression("B", 2)
    ten = irg.NodeConstant(10)
    edge_b_ten = irg.Edge(indices_offsets=[0, 0])
    graph = nx.MultiDiGraph()
    graph.add_edge(b, ten, key=edge_b_ten)

    stage_src = t2pdb.translate_graph(graph)

    assert stage_src == ["B[i,j] = 10"]


def test_assignment():
    """
    Definitions:
    B = A
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    edge_ba = irg.Edge(indices_offsets=[0, 0])
    graph = nx.MultiDiGraph()
    graph.add_edge(b, a, key=edge_ba)

    stage_src = t2pdb.translate_graph(graph)

    assert stage_src == ["B[i,j] = A[i,j]"]


def test_binary_operator():
    """
    Definitions: A = B + C
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    plus = irg.NodeBinaryOperator("+")
    edge_a_plus = irg.Edge(indices_offsets=[0, 0])
    edge_plus_b = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
    edge_plus_c = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)

    graph = nx.MultiDiGraph()
    graph.add_edge(a, plus, key=edge_a_plus)
    graph.add_edge(plus, b, key=edge_plus_b)
    graph.add_edge(plus, c, key=edge_plus_c)

    stage_src = t2pdb.translate_graph(graph)

    assert stage_src == ["A[i,j] = (B[i,j]) + (C[i,j])"]


def test_indices_offsets():
    """
    Definitions: A[i, j] = B[i-1, j+1] + C[i+1, j-1]
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    plus = irg.NodeBinaryOperator("+")
    edge_a_plus = irg.Edge(indices_offsets=[0, 0])
    edge_plus_b = irg.Edge(indices_offsets=[-1, +1], is_left_edge=True)
    edge_plus_c = irg.Edge(indices_offsets=[+1, -1], is_right_edge=True)

    graph = nx.MultiDiGraph()
    graph.add_edge(a, plus, key=edge_a_plus)
    graph.add_edge(plus, b, key=edge_plus_b)
    graph.add_edge(plus, c, key=edge_plus_c)

    stage_src = t2pdb.translate_graph(graph)

    assert stage_src == ["A[i,j] = (B[i-1,j+1]) + (C[i+1,j-1])"]


@pytest.fixture(scope="module")
def dummy_graph():
    """
    A simple graph representing the expressions

        out[p] = A[p] + 10
        out_2[p] = A[p] + 10
        out_3[p] = B[p + (-1,0,0)] - C[p + (1,1,0)]

    It has 3 roots, 4 leaves and 2 connected components
    """
    out_node = irg.NodeNamedExpression("out", 2)
    a_node = irg.NodeNamedExpression("A", 2)
    const_node = irg.NodeConstant(10)
    a_edge = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
    const_edge = irg.Edge(indices_offsets=None, is_right_edge=True)
    plus_node = irg.NodeBinaryOperator("+")

    out2_node = irg.NodeNamedExpression("out_2", 2)

    out3_node = irg.NodeNamedExpression("out_3", 2)
    b_node = irg.NodeNamedExpression("B", 2)
    c_node = irg.NodeNamedExpression("C", 2)
    b_edge = irg.Edge(indices_offsets=[-1, 0], is_left_edge=True)
    c_edge = irg.Edge(indices_offsets=[1, 1], is_right_edge=True)
    minus_node = irg.NodeBinaryOperator("-")

    graph = nx.MultiDiGraph()
    graph.add_edge(out_node, plus_node)
    graph.add_edge(plus_node, a_node, key=a_edge)
    graph.add_edge(plus_node, const_node, key=const_edge)
    graph.add_edge(out2_node, plus_node)
    graph.add_edge(out3_node, minus_node)
    graph.add_edge(minus_node, b_node, key=b_edge)
    graph.add_edge(minus_node, c_node, key=c_edge)

    return graph


def test_translate_graph_returns_outputs(dummy_graph):
    stage_src = t2pdb.translate_graph(dummy_graph)
    assert set(stage_src) == {"out[i,j] = (A[i,j]) + (10)",
                              "out_2[i,j] = (A[i,j]) + (10)",
                              "out_3[i,j] = (B[i-1,j]) - (C[i+1,j+1])"}
