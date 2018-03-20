import pytest
import networkx as nx

import gridtools.intermediate_representation.utils as irutils
from gridtools.intermediate_representation.ir import IR
import gridtools.intermediate_representation.graph as irg
import gridtools.backend.assignment_statements_detection as asd


def test_empty_graph():
    """
    Definitions: {} (no definitions)
    """
    graph = nx.MultiDiGraph()
    expected_graphs = []
    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_single_node():
    """
    Definitions: A
    """
    a = irg.NodeNamedExpression("A", 2)
    graph = nx.MultiDiGraph()
    graph.add_node(a)

    expected_graphs = [graph.copy()]

    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_assignment():
    """
    Definitions: A = B
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    edge = irg.Edge(indices_offsets=None)

    graph = nx.MultiDiGraph()
    graph.add_edge(a, b, key=edge)
    expected_graphs = [graph.copy()]

    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_binary_operator():
    """
    Definitions: A = B + C
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    plus = irg.NodeBinaryOperator("+")
    edge_a_plus = irg.Edge(indices_offsets=[0, 0])
    edge_plus_b = irg.Edge(indices_offsets=[0, 1], is_left_edge=True)
    edge_plus_c = irg.Edge(indices_offsets=[1, 0], is_right_edge=True)

    graph = nx.MultiDiGraph()
    graph.add_edge(a, plus, key=edge_a_plus)
    graph.add_edge(plus, b, key=edge_plus_b)
    graph.add_edge(plus, c, key=edge_plus_c)

    expected_graphs = [graph.copy()]

    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_two_assignment_statements():
    """
    Definitions:
    C = A + B
    D = C - 10
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    d = irg.NodeNamedExpression("D", 2)
    ten = irg.NodeConstant(10)
    plus = irg.NodeBinaryOperator("+")
    minus = irg.NodeBinaryOperator("-")
    edge_c_plus = irg.Edge(indices_offsets=[0, 0])
    edge_plus_a = irg.Edge(indices_offsets=[0, 1], is_left_edge=True)
    edge_plus_b = irg.Edge(indices_offsets=[1, 0], is_right_edge=True)
    edge_d_minus = irg.Edge(indices_offsets=[1, 1])
    edge_minus_c = irg.Edge(indices_offsets=[0, 2], is_left_edge=True)
    edge_minus_ten = irg.Edge(indices_offsets=[2, 0], is_right_edge=True)

    graph = nx.MultiDiGraph()
    graph.add_edge(c, plus, key=edge_c_plus)
    graph.add_edge(plus, a, key=edge_plus_a)
    graph.add_edge(plus, b, key=edge_plus_b)
    graph.add_edge(d, minus, key=edge_d_minus)
    graph.add_edge(minus, c, key=edge_minus_c)
    graph.add_edge(minus, ten, key=edge_minus_ten)

    expected_graph_c = graph.subgraph([c, a, plus, b])
    expected_graph_d = graph.subgraph([d, c, minus, ten])
    expected_graphs = [expected_graph_c, expected_graph_d]

    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_temporary():
    """
    Definitions:
    B = A   <-- B is the temporary
    C = B
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    edge_ba = irg.Edge(indices_offsets=[0, 0])
    edge_cb = irg.Edge(indices_offsets=[0, 1])

    graph = nx.MultiDiGraph()
    graph.add_edge(b, a, key=edge_ba)
    graph.add_edge(c, b, key=edge_cb)

    expected_graph_b = graph.subgraph([b, a])
    expected_graph_c = graph.subgraph([c, b])
    expected_graphs = [expected_graph_b, expected_graph_c]

    _assert_generated_graphs_are_correct(graph, expected_graphs)


def test_two_components():
    """
    Definitions:
    C = A + B   <-- forms one graph's component
    F = D - E   <-- forms another graph's component
    """
    a = irg.NodeNamedExpression("A", 2)
    b = irg.NodeNamedExpression("B", 2)
    c = irg.NodeNamedExpression("C", 2)
    d = irg.NodeNamedExpression("D", 2)
    e = irg.NodeNamedExpression("E", 2)
    f = irg.NodeNamedExpression("F", 2)
    plus = irg.NodeBinaryOperator("+")
    minus = irg.NodeBinaryOperator("-")
    edge_c_plus = irg.Edge(indices_offsets=[0, 0])
    edge_plus_a = irg.Edge(indices_offsets=[1, 0], is_left_edge=True)
    edge_plus_b = irg.Edge(indices_offsets=[2, 0], is_right_edge=True)
    edge_f_minus = irg.Edge(indices_offsets=[3, 0])
    edge_minus_d = irg.Edge(indices_offsets=[4, 0], is_left_edge=True)
    edge_minus_e = irg.Edge(indices_offsets=[5, 0], is_right_edge=True)

    graph = nx.MultiDiGraph()
    graph.add_edge(c, plus, key=edge_c_plus)
    graph.add_edge(plus, a, key=edge_plus_a)
    graph.add_edge(plus, b, key=edge_plus_b)
    graph.add_edge(f, minus, key=edge_f_minus)
    graph.add_edge(minus, d, key=edge_minus_d)
    graph.add_edge(minus, e, key=edge_minus_e)

    expected_graph_c = graph.subgraph([c, a, plus, b])
    expected_graph_f = graph.subgraph([f, d, minus, e])
    expected_graphs = [expected_graph_c, expected_graph_f]

    actual_graphs = _generate_assignment_statements_graphs(graph)

    actual_graphs_by_root_name = {}
    for g in actual_graphs:
        assert len(irutils.find_roots(g)) == 1
        root_name = str(irutils.find_roots(g)[0])
        assert root_name not in actual_graphs_by_root_name
        actual_graphs_by_root_name[root_name] = g

    for expected_graph in expected_graphs:
        root_name = str(irutils.find_roots(expected_graph)[0])
        actual_graph = actual_graphs_by_root_name[root_name]
        irutils.assert_graphs_are_equal(actual_graph, expected_graph)


def _assert_generated_graphs_are_correct(graph, expected_graphs):
    actual_graphs = _generate_assignment_statements_graphs(graph)
    assert len(actual_graphs) == len(expected_graphs)
    for expected_graph, actual_graph in zip(expected_graphs, actual_graphs):
        irutils.assert_graphs_are_equal(expected_graph, actual_graph)


def _generate_assignment_statements_graphs(graph):
    ir = IR(stencil_configs=None, graph=graph)
    detection = asd.AssignmentStatementsDetection()
    ir = detection.process(ir)
    graphs = ir.assignment_statements_graphs
    return graphs


def test_find_topologically_sorted_temporaries(hdg, gtdd):
    temps = asd.find_topologically_sorted_temporaries(hdg.G)
    assert type(temps) == list
    assert set(temps) == hdg.temps

    temps = asd.find_topologically_sorted_temporaries(gtdd.G)
    assert type(temps) == list
    assert set(temps) == gtdd.temps


def test_dfs_assignment_statements(hdg, gtdd):
    for stage_name, headless_subgraph in hdg.headless_stages.items():
        result = asd.dfs_assignment_statements(hdg.stage_node_lists[stage_name][1],
                                               hdg.headless_stages[stage_name])
        assert type(result) == list
        assert set(result) == set(hdg.stage_node_lists[stage_name][1:])

    for stage_name, headless_subgraph in gtdd.headless_stages.items():
        result = asd.dfs_assignment_statements(gtdd.stage_node_lists[stage_name][1],
                                               gtdd.headless_stages[stage_name])
        assert type(result) == list
        assert set(result) == set(gtdd.stage_node_lists[stage_name][1:])


def test_start_dfs_assignment_statements(hdg, gtdd):
    for stage_name, stage_subgraph in hdg.stages.items():
        result = asd.start_dfs_assignment_statements(hdg.stage_node_lists[stage_name][0],
                                                     hdg.stages[stage_name])
        assert type(result) == list
        assert set(result) == set(hdg.stage_node_lists[stage_name])

    for stage_name, stage_subgraph in gtdd.stages.items():
        result = asd.start_dfs_assignment_statements(gtdd.stage_node_lists[stage_name][0],
                                                     gtdd.stages[stage_name])
        assert type(result) == list
        assert set(result) == set(gtdd.stage_node_lists[stage_name])


def test_find_topologically_sorted_assignment_statements(hdg, gtdd):
    detected_stages = asd.find_topologically_sorted_assignment_statements(hdg.G)
    assert len(detected_stages) == len(hdg.stages)
    for stg in detected_stages:
        output_name = irutils.find_roots(stg)[0].name
        irutils.assert_graphs_are_equal(stg, hdg.stages[output_name])

    detected_stages = asd.find_topologically_sorted_assignment_statements(gtdd.G)
    assert len(detected_stages) == len(gtdd.stages)
    for i, stg in enumerate(detected_stages):
        output_name = irutils.find_roots(stg)[0].name
        irutils.assert_graphs_are_equal(stg, gtdd.stages[output_name])
