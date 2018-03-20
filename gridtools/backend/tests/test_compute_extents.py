import pytest
import numpy as np
import networkx as nx

import gridtools.intermediate_representation.utils as irutils
import gridtools.intermediate_representation.graph as irg
import gridtools.backend.compute_extents as cext


class DummyGraph():
    """
    A simple graph representing the expressions
        A[i, j] = B[i+1, j] + C[i, j+1]
        D[i, j] = A[i-1, j-1] + 10
        E[i, j] = B[i+1, j+1]
    """
    def __init__(self):
        self.a_node = irg.NodeNamedExpression("A", 2)
        self.b_node = irg.NodeNamedExpression("B", 2)
        self.c_node = irg.NodeNamedExpression("C", 2)
        self.d_node = irg.NodeNamedExpression("D", 2)
        self.e_node = irg.NodeNamedExpression("E", 2)
        self.const_node = irg.NodeConstant(10)
        self.plus1_node = irg.NodeBinaryOperator("+")
        self.plus2_node = irg.NodeBinaryOperator("+")

        self.edge_a_plus1 = irg.Edge(indices_offsets=[0, 0])
        self.edge_plus1_b = irg.Edge(indices_offsets=[1, 0], is_right_edge=True)
        self.edge_plus1_c = irg.Edge(indices_offsets=[0, 1], is_left_edge=True)
        self.edge_d_plus2 = irg.Edge(indices_offsets=[0, 0])
        self.edge_plus2_a = irg.Edge(indices_offsets=[-1, -1], is_right_edge=True)
        self.edge_plus2_const = irg.Edge(indices_offsets=[0,0], is_left_edge=True)
        self.edge_e_b = irg.Edge(indices_offsets=[1, 1])

        self.G = nx.MultiDiGraph()
        self.G.add_edge(self.a_node, self.plus1_node, key=self.edge_a_plus1)
        self.G.add_edge(self.plus1_node, self.b_node, key=self.edge_plus1_b)
        self.G.add_edge(self.plus1_node, self.c_node, key=self.edge_plus1_c)
        self.G.add_edge(self.d_node, self.plus2_node, key=self.edge_d_plus2)
        self.G.add_edge(self.plus2_node, self.a_node, key=self.edge_plus2_a)
        self.G.add_edge(self.plus2_node, self.const_node, key=self.edge_plus2_const)
        self.G.add_edge(self.e_node, self.b_node, key=self.edge_e_b)

        # Reference result for tests
        self.minimum_halo = np.array([1, 1, 1, 1])


# This is not a module-scope fixture because the graph is modified in-place, thus
# every test needs a fresh new graph to work with
@pytest.fixture
def dummy_graph():
    return DummyGraph()


def test_convert_offset_to_extent():
    aec = cext.AccessExtentsComputation()
    assert np.array_equal(aec._convert_offsets_to_extent_format([0, 0]), np.array([0, 0, 0, 0]))
    assert np.array_equal(aec._convert_offsets_to_extent_format([1, 0]), np.array([0, 1, 0, 0]))
    assert np.array_equal(aec._convert_offsets_to_extent_format([0, 1]), np.array([0, 0, 0, 1]))
    assert np.array_equal(aec._convert_offsets_to_extent_format([-1, 0]), np.array([1, 0, 0, 0]))
    assert np.array_equal(aec._convert_offsets_to_extent_format([0, -1]), np.array([0, 0, 1, 0]))
    assert np.array_equal(aec._convert_offsets_to_extent_format([2, -2]), np.array([0, 2, 2, 0]))


def test_dfs_access_extents_with_constant():
    aec = cext.AccessExtentsComputation()
    const_node = irg.NodeConstant(1.0)
    graph = nx.MultiDiGraph()
    graph.add_node(const_node)
    assert aec._dfs_access_extents(const_node, graph, np.array([0, 0, 0, 0])) is None
    assert hasattr(const_node, "access_extent") is False


def test_dfs_access_extents_with_named_expression_leaf(dummy_graph):
    aec = cext.AccessExtentsComputation()
    assert aec._dfs_access_extents(dummy_graph.b_node,
                                   dummy_graph.G,
                                   np.array([1, 1, 1, 1])) is None
    assert aec._dfs_access_extents(dummy_graph.c_node,
                                   dummy_graph.G,
                                   np.array([0, 0, 0, 0])) is None
    assert np.array_equal(dummy_graph.b_node.access_extent, np.array([1, 1, 1, 1]))
    assert np.array_equal(dummy_graph.c_node.access_extent, np.array([0, 0, 0, 0]))


def test_dfs_access_extents_with_binary_operator(dummy_graph):
    aec = cext.AccessExtentsComputation()
    assert aec._dfs_access_extents(dummy_graph.plus1_node,
                                   dummy_graph.G,
                                   np.array([0, 0, 0, 0])) is None
    assert np.array_equal(dummy_graph.b_node.access_extent, np.array([0, 1, 0, 0]))
    assert np.array_equal(dummy_graph.c_node.access_extent, np.array([0, 0, 0, 1]))
    assert hasattr(dummy_graph.plus1_node, "access_extent") is False


def test_dfs_access_extents_with_named_expression_non_leaf(dummy_graph):
    aec = cext.AccessExtentsComputation()
    assert aec._dfs_access_extents(dummy_graph.a_node,
                                   dummy_graph.G,
                                   np.array([0, 0, 0, 0])) is None
    assert np.array_equal(dummy_graph.a_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(dummy_graph.b_node.access_extent, np.array([0, 1, 0, 0]))
    assert np.array_equal(dummy_graph.c_node.access_extent, np.array([0, 0, 0, 1]))
    assert hasattr(dummy_graph.plus1_node, "access_extent") is False


def test_dfs_access_extents_with_simple_assignment(dummy_graph):
    aec = cext.AccessExtentsComputation()
    assert aec._dfs_access_extents(dummy_graph.e_node,
                                   dummy_graph.G,
                                   np.array([0, 0, 0, 0])) is None
    assert np.array_equal(dummy_graph.e_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(dummy_graph.b_node.access_extent, np.array([0, 1, 0, 1]))


def test_compute_access_extents(dummy_graph, hdg, gtdd):
    aec = cext.AccessExtentsComputation()

    # Test with dummy graph
    aec._compute_access_extents(dummy_graph.G)
    assert np.array_equal(dummy_graph.e_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(dummy_graph.d_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(dummy_graph.a_node.access_extent, np.array([1, 0, 1, 0]))
    assert np.array_equal(dummy_graph.b_node.access_extent, np.array([1, 1, 1, 1]))
    assert np.array_equal(dummy_graph.c_node.access_extent, np.array([1, 0, 1, 1]))

    # Test with horizontal diffusion
    aec._compute_access_extents(hdg.G)
    assert np.array_equal(hdg.diffusion_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(hdg.wgt_node.access_extent, np.array([0, 0, 0, 0]))
    assert np.array_equal(hdg.fli_node.access_extent, np.array([1, 0, 0, 0]))
    assert np.array_equal(hdg.flj_node.access_extent, np.array([0, 0, 1, 0]))
    assert np.array_equal(hdg.lap_node.access_extent, np.array([1, 1, 1, 1]))
    assert np.array_equal(hdg.data_node.access_extent, np.array([2, 2, 2, 2]))

    # Test with GridTools Data Dependency example
    aec._compute_access_extents(gtdd.G)
    assert np.array_equal(gtdd.a_node.access_extent, np.array([1, 2, 0, 0]))
    assert np.array_equal(gtdd.b_node.access_extent, np.array([4, 3, 0, 0]))
    assert np.array_equal(gtdd.c_node.access_extent, np.array([3, 4, 0, 0]))
    assert np.array_equal(gtdd.d_node.access_extent, np.array([2, 2, 0, 0]))
    assert np.array_equal(gtdd.e_node.access_extent, np.array([0, 0, 0, 0]))


def test_compute_minimum_halo(dummy_graph, hdg, gtdd):
    aec = cext.AccessExtentsComputation()

    # Test with dummy graph
    aec._compute_access_extents(dummy_graph.G)
    assert np.array_equal(aec._compute_minimum_halo(dummy_graph.G), np.array([1, 1, 1, 1]))

    # Test with horizontal diffusion
    aec._compute_access_extents(hdg.G)
    assert np.array_equal(aec._compute_minimum_halo(hdg.G), hdg.minimum_halo)

    # Test with GridTools Data Dependency example
    aec._compute_access_extents(gtdd.G)
    assert np.array_equal(aec._compute_minimum_halo(gtdd.G), gtdd.minimum_halo)
