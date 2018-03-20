import networkx as nx

from gridtools.user_interface.ngstencil import NGStencilConfigs
import gridtools.intermediate_representation.utils as irutils
from gridtools.intermediate_representation import graph
from gridtools.frontend.crappy.expression import Equation
from gridtools.frontend.crappy.frontend import Frontend
from gridtools.frontend.crappy.index import Index

frontend = Frontend()


def test_assignment():
    expected_graph = nx.MultiDiGraph()
    expected_graph.add_edge(graph.NodeNamedExpression("A", 2),
                            graph.NodeNamedExpression("data", 2),
                            key=graph.Edge(indices_offsets=[0, 0]))
    _test_frontend(definitions_assignment, expected_graph)


def definitions_assignment(data):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data[i, j]
    return A


def test_two_assignments():
    expected_graph = nx.MultiDiGraph()
    node_data = graph.NodeNamedExpression("data", 2)
    node_b = graph.NodeNamedExpression("B", 2)
    node_a = graph.NodeNamedExpression("A", 2)

    expected_graph.add_edge(node_b, node_a, key=graph.Edge(indices_offsets=[0, 0]))
    expected_graph.add_edge(node_a, node_data, key=graph.Edge(indices_offsets=[0, 0]))

    _test_frontend(definitions_two_assignments, expected_graph)


def definitions_two_assignments(data):
    i = Index()
    j = Index()
    A = Equation()
    B = Equation()
    A[i, j] = data[i, j]
    B[i, j] = A[i, j]
    return B


def test_two_inputs():
    node_a = graph.NodeNamedExpression("A", 2)
    node_plus = graph.NodeBinaryOperator("+")
    node_data0 = graph.NodeNamedExpression("data0", 2)
    node_data1 = graph.NodeNamedExpression("data1", 2)

    expected_graph = nx.MultiDiGraph()
    expected_graph.add_edge(node_a, node_plus, key=graph.Edge(indices_offsets=[0, 0]))
    expected_graph.add_edge(node_plus, node_data0, key=graph.Edge(indices_offsets=[0, 0], is_left_edge=True))
    expected_graph.add_edge(node_plus, node_data1, key=graph.Edge(indices_offsets=[0, 0], is_right_edge=True))

    _test_frontend(definitions_two_inputs, expected_graph)


def definitions_two_inputs(data0, data1):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data0[i, j] + data1[i, j]
    return A


def test_binary_operators():
    _test_binary_operator(definitions_binary_operator_add, "+")
    _test_binary_operator(definitions_binary_operator_sub, "-")
    _test_binary_operator(definitions_binary_operator_mul, "*")
    _test_binary_operator(definitions_binary_operator_truediv, "/")


def definitions_binary_operator_add(data):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data[i, j] + data[i, j]
    return A


def definitions_binary_operator_sub(data):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data[i, j] - data[i, j]
    return A


def definitions_binary_operator_mul(data):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data[i, j] * data[i, j]
    return A


def definitions_binary_operator_truediv(data):
    i = Index()
    j = Index()
    A = Equation()
    A[i, j] = data[i, j] / data[i, j]
    return A


def _test_binary_operator(definitions_func, operator_symbol):
    expected_graph = nx.MultiDiGraph()
    node_a = graph.NodeNamedExpression("A", 2)
    node_plus = graph.NodeBinaryOperator(operator_symbol)
    node_data = graph.NodeNamedExpression("data", 2)

    expected_graph.add_edge(node_a, node_plus, key=graph.Edge(indices_offsets=[0, 0]))
    expected_graph.add_edge(node_plus, node_data, key=graph.Edge(indices_offsets=[0, 0], is_left_edge=True))
    expected_graph.add_edge(node_plus, node_data, key=graph.Edge(indices_offsets=[0, 0], is_right_edge=True))

    _test_frontend(definitions_func, expected_graph)


def test_indices_offsets():
    expected_graph = nx.MultiDiGraph()
    node_c = graph.NodeNamedExpression("C", 2)
    node_b = graph.NodeNamedExpression("B", 2)
    node_a = graph.NodeNamedExpression("A", 2)
    node_data = graph.NodeNamedExpression("data", 2)

    expected_graph.add_edge(node_c, node_b, key=graph.Edge(indices_offsets=[-10, +11]))
    expected_graph.add_edge(node_b, node_a, key=graph.Edge(indices_offsets=[+10, -11]))
    expected_graph.add_edge(node_a, node_data, key=graph.Edge(indices_offsets=[0, 0]))

    _test_frontend(definitions_indices_offsets, expected_graph)


def definitions_indices_offsets(data):
    i = Index()
    j = Index()
    A = Equation()
    B = Equation()
    C = Equation()
    A[i, j] = data[i, j]
    B[i, j] = A[i+10, j-11]
    C[i, j] = B[i-10, j+11]
    return C


def _test_frontend(definitions_func, expected_graph):
    stencil_configs = NGStencilConfigs(definitions_func=definitions_func,
                                       inputs=None,
                                       outputs=None,
                                       domain=None,
                                       mode=None)
    ir = frontend.process(stencil_configs)
    actual_graph = ir.graph
    irutils.assert_graphs_are_equal(actual_graph, expected_graph)
