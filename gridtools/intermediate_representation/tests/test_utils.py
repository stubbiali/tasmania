import os

import pytest
import networkx as nx
import matplotlib.pyplot as plt

import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils


class DummyGraph():
    """
    A simple graph representing the expressions

        out[p] = A[p] + 10
        out_2[p] = A[p] + 10
        out_3[p] = B[p + (-1,0,0)] - C[p + (1,1,0)]

    It has 3 roots, 4 leaves and 2 connected components
    """
    def __init__(self):
        self.out_node = irg.NodeNamedExpression("out", 2)
        self.a_node = irg.NodeNamedExpression("A", 2)
        self.const_node = irg.NodeConstant(10)
        self.a_edge = irg.Edge(indices_offsets=[0, 0])
        self.const_edge = irg.Edge(indices_offsets=None)
        self.plus_node = irg.NodeBinaryOperator("+")

        self.out2_node = irg.NodeNamedExpression("out_2", 2)

        self.out3_node = irg.NodeNamedExpression("out_3", 2)
        self.b_node = irg.NodeNamedExpression("B", 2)
        self.c_node = irg.NodeNamedExpression("C", 2)
        self.b_edge = irg.Edge(indices_offsets=[-1, 0])
        self.c_edge = irg.Edge(indices_offsets=[1, 1])
        self.minus_node = irg.NodeBinaryOperator("-")

        self.G = nx.MultiDiGraph()
        self.G.add_edge(self.out_node, self.plus_node)
        self.G.add_edge(self.plus_node, self.a_node, key="left", data=self.a_edge)
        self.G.add_edge(self.plus_node, self.const_node, key="right", data=self.const_edge)
        self.G.add_edge(self.out2_node, self.plus_node)
        self.G.add_edge(self.out3_node, self.minus_node)
        self.G.add_edge(self.minus_node, self.b_node, key="left", data=self.b_edge)
        self.G.add_edge(self.minus_node, self.c_node, key="right", data=self.c_edge)

        self.roots = set((self.out_node, self.out2_node, self.out3_node))
        self.leaves = set((self.a_node, self.const_node, self.b_node, self.c_node))


@pytest.fixture(scope="module")
def dummy_graph():
    return DummyGraph()


def test_find_roots(dummy_graph):
    graph_roots = irutils.find_roots(dummy_graph.G)
    assert type(graph_roots) == list
    assert set(graph_roots) == dummy_graph.roots


def test_find_leaves(dummy_graph):
    graph_leaves = irutils.find_leaves(dummy_graph.G)
    assert type(graph_leaves) == list
    assert set(graph_leaves) == dummy_graph.leaves


@pytest.mark.skipif(os.environ.get('DISPLAY') is None,
                    reason="could not detect DISPLAY environment variable")
def test_plot_graph_doesnt_crash(dummy_graph):
    plt.ion()
    irutils.plot_graph(dummy_graph.G)
    plt.close('all')
    plt.ioff()


def test_save_graph_produces_output(dummy_graph):
    irutils.save_graph(dummy_graph.G, 'test_graph')
    assert os.path.isfile('./test_graph.svg')
    assert os.path.isfile('./test_graph')
    os.remove('./test_graph.svg')
    os.remove('./test_graph')
