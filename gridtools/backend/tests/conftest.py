import pytest
import numpy as np
import networkx as nx

import gridtools.intermediate_representation.utils as irutils
import gridtools.intermediate_representation.graph as irg


class HorizontalDiffusionGraph:
    """
    An intermediate representation graph of an Horizontal Diffusion stencil
    """
    def __init__(self):
        # Named Expression Nodes
        self.diffusion_node = irg.NodeNamedExpression("diffusion", 2)
        self.wgt_node = irg.NodeNamedExpression("wgt", 2)
        self.fli_node = irg.NodeNamedExpression("flux_i", 2)
        self.flj_node = irg.NodeNamedExpression("flux_j", 2)
        self.lap_node = irg.NodeNamedExpression("laplace", 2)
        self.data_node = irg.NodeNamedExpression("data", 2)

        # Constant node
        self.alpha_node = irg.NodeConstant(-4.0)

        # Binary Operator Nodes
        self.mult1_node = irg.NodeBinaryOperator("*")
        self.mult2_node = irg.NodeBinaryOperator("*")
        self.minus1_node = irg.NodeBinaryOperator("-")
        self.minus2_node = irg.NodeBinaryOperator("-")
        self.minus3_node = irg.NodeBinaryOperator("-")
        self.minus4_node = irg.NodeBinaryOperator("-")
        self.plus1_node = irg.NodeBinaryOperator("+")
        self.plus2_node = irg.NodeBinaryOperator("+")
        self.plus3_node = irg.NodeBinaryOperator("+")
        self.plus4_node = irg.NodeBinaryOperator("+")
        self.plus5_node = irg.NodeBinaryOperator("+")

        # Diffusion expression edges
        self.diffusion_to_mul1_edge = irg.Edge(indices_offsets=[0, 0])
        self.mul1_to_wgt_edge = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.mul1_to_min1_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.min1_to_fli_edge = irg.Edge(indices_offsets=[-1, 0], is_left_edge=True)
        self.min1_to_plu1_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.plu1_to_fli_edge = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.plu1_to_min2_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.min2_to_flj_left_edge = irg.Edge(indices_offsets=[0, -1], is_left_edge=True)
        self.min2_to_flj_right_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)

        # Flux J expression edges
        self.flj_to_min3_edge = irg.Edge(indices_offsets=[0, 0])
        self.min3_to_lap_left_edge = irg.Edge(indices_offsets=[0, 1], is_left_edge=True)
        self.min3_to_lap_right_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)

        # Flux I expression edges
        self.fli_to_min4_edge = irg.Edge(indices_offsets=[0, 0])
        self.min4_to_lap_left_edge = irg.Edge(indices_offsets=[1, 0], is_left_edge=True)
        self.min4_to_lap_right_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)

        # Laplacian expression edges
        self.lap_to_mul2_edge = irg.Edge(indices_offsets=[0, 0])
        self.mul2_to_alpha_edge = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.mul2_to_plu2_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.plu2_to_data_edge = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.plu2_to_plu3_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.plu3_to_data_edge = irg.Edge(indices_offsets=[-1, 0], is_left_edge=True)
        self.plu3_to_plu4_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.plu4_to_data_edge = irg.Edge(indices_offsets=[1, 0], is_left_edge=True)
        self.plu4_to_plu5_edge = irg.Edge(indices_offsets=[0, 0], is_right_edge=True)
        self.plu5_to_data_left_edge = irg.Edge(indices_offsets=[0, -1], is_left_edge=True)
        self.plu5_to_data_right_edge = irg.Edge(indices_offsets=[0, 1], is_right_edge=True)

        # Graph assembly
        self.G = nx.MultiDiGraph()

        self.G.add_edge(self.diffusion_node, self.mult1_node, key=self.diffusion_to_mul1_edge)
        self.G.add_edge(self.mult1_node, self.wgt_node, key=self.mul1_to_wgt_edge)
        self.G.add_edge(self.mult1_node, self.minus1_node, key=self.mul1_to_min1_edge)
        self.G.add_edge(self.minus1_node, self.fli_node, key=self.min1_to_fli_edge)
        self.G.add_edge(self.minus1_node, self.plus1_node, key=self.min1_to_plu1_edge)
        self.G.add_edge(self.plus1_node, self.fli_node, key=self.plu1_to_fli_edge)
        self.G.add_edge(self.plus1_node, self.minus2_node, key=self.plu1_to_min2_edge)
        self.G.add_edge(self.minus2_node, self.flj_node, key=self.min2_to_flj_left_edge)
        self.G.add_edge(self.minus2_node, self.flj_node, key=self.min2_to_flj_right_edge)

        self.G.add_edge(self.flj_node, self.minus3_node, key=self.flj_to_min3_edge)
        self.G.add_edge(self.minus3_node, self.lap_node, key=self.min3_to_lap_left_edge)
        self.G.add_edge(self.minus3_node, self.lap_node, key=self.min3_to_lap_right_edge)

        self.G.add_edge(self.fli_node, self.minus4_node, key=self.fli_to_min4_edge)
        self.G.add_edge(self.minus4_node, self.lap_node, key=self.min4_to_lap_left_edge)
        self.G.add_edge(self.minus4_node, self.lap_node, key=self.min4_to_lap_right_edge)

        self.G.add_edge(self.lap_node, self.mult2_node, key=self.lap_to_mul2_edge)
        self.G.add_edge(self.mult2_node, self.alpha_node, key=self.mul2_to_alpha_edge)
        self.G.add_edge(self.mult2_node, self.plus2_node, key=self.mul2_to_plu2_edge)
        self.G.add_edge(self.plus2_node, self.data_node, key=self.plu2_to_data_edge)
        self.G.add_edge(self.plus2_node, self.plus3_node, key=self.plu2_to_plu3_edge)
        self.G.add_edge(self.plus3_node, self.data_node, key=self.plu3_to_data_edge)
        self.G.add_edge(self.plus3_node, self.plus4_node, key=self.plu3_to_plu4_edge)
        self.G.add_edge(self.plus4_node, self.data_node, key=self.plu4_to_data_edge)
        self.G.add_edge(self.plus4_node, self.plus5_node, key=self.plu4_to_plu5_edge)
        self.G.add_edge(self.plus5_node, self.data_node, key=self.plu5_to_data_left_edge)
        self.G.add_edge(self.plus5_node, self.data_node, key=self.plu5_to_data_right_edge)

        # Test results for this graph
        self.temps = {self.fli_node, self.flj_node, self.lap_node}
        self.stage_node_lists = {'diffusion': [self.diffusion_node,
                                               self.mult1_node,
                                               self.wgt_node,
                                               self.minus1_node,
                                               self.plus1_node,
                                               self.fli_node,
                                               self.minus2_node,
                                               self.flj_node],
                                 'flux_j': [self.flj_node,
                                            self.minus3_node,
                                            self.lap_node],
                                 'flux_i': [self.fli_node,
                                            self.minus4_node,
                                            self.lap_node],
                                 'laplace': [self.lap_node,
                                             self.mult2_node,
                                             self.alpha_node,
                                             self.plus2_node,
                                             self.data_node,
                                             self.plus3_node,
                                             self.plus4_node,
                                             self.plus5_node]
                                 }
        self.stages = {name: self.G.subgraph(node_list) for name, node_list in self.stage_node_lists.items()}
        self.headless_stages = {name: self.G.subgraph(node_list[1:]) for name, node_list in self.stage_node_lists.items()}
        self.minimum_halo = np.array([2, 2, 2, 2])


@pytest.fixture(scope="module")
def hdg():
    return HorizontalDiffusionGraph()


class GridToolsDataDepsGraph():
    """
    An intermediate representation graph of the first example from the Data
    Dependency Analysis page in the GridTools wiki:
    https://github.com/eth-cscs/gridtools/wiki/Data-Dependencies-Analysis-in-GridTools
    """
    def __init__(self):
        # Named Expression Nodes
        self.a_node = irg.NodeNamedExpression("a", 2)
        self.b_node = irg.NodeNamedExpression("b", 2)
        self.c_node = irg.NodeNamedExpression("c", 2)
        self.d_node = irg.NodeNamedExpression("d", 2)
        self.e_node = irg.NodeNamedExpression("e", 2)

        # Binary Operator Nodes
        self.plus1_node = irg.NodeBinaryOperator("+")
        self.plus2_node = irg.NodeBinaryOperator("+")
        self.plus3_node = irg.NodeBinaryOperator("+")
        self.plus4_node = irg.NodeBinaryOperator("+")
        self.plus5_node = irg.NodeBinaryOperator("+")
        self.plus6_node = irg.NodeBinaryOperator("+")
        self.plus7_node = irg.NodeBinaryOperator("+")
        self.plus8_node = irg.NodeBinaryOperator("+")
        self.plus9_node = irg.NodeBinaryOperator("+")

        # F2 stage edges
        self.edge_e_p1 = irg.Edge(indices_offsets=[0, 0])
        self.edge_p1_a = irg.Edge(indices_offsets=[-1, 0], is_right_edge=True)
        self.edge_p1_p2 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p2_a = irg.Edge(indices_offsets=[2, 0], is_right_edge=True)
        self.edge_p2_p3 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p3_d = irg.Edge(indices_offsets=[-2, 0], is_right_edge=True)
        self.edge_p3_p4 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p4_d = irg.Edge(indices_offsets=[2, 0], is_right_edge=True)
        self.edge_p4_p5 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p5_c_left = irg.Edge(indices_offsets=[-1, 0], is_right_edge=True)
        self.edge_p5_c_right = irg.Edge(indices_offsets=[1, 0], is_left_edge=True)

        # F1 stage edges
        self.edge_d_p6 = irg.Edge(indices_offsets=[0, 0])
        self.edge_p6_b = irg.Edge(indices_offsets=[-2, 0], is_right_edge=True)
        self.edge_p6_p7 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p7_c_left = irg.Edge(indices_offsets=[-1, 0], is_right_edge=True)
        self.edge_p7_c_right = irg.Edge(indices_offsets=[2, 0], is_left_edge=True)

        # F0 stage edges
        self.edge_a_p8 = irg.Edge(indices_offsets=[0, 0])
        self.edge_p8_b = irg.Edge(indices_offsets=[-1, 0], is_right_edge=True)
        self.edge_p8_p9 = irg.Edge(indices_offsets=[0, 0], is_left_edge=True)
        self.edge_p9_b = irg.Edge(indices_offsets=[1, 0], is_right_edge=True)
        self.edge_p9_c = irg.Edge(indices_offsets=[1, 0], is_left_edge=True)

        # Graph assembly
        self.G = nx.MultiDiGraph()

        self.G.add_edge(self.e_node, self.plus1_node, key=self.edge_e_p1)
        self.G.add_edge(self.plus1_node, self.a_node, key=self.edge_p1_a)
        self.G.add_edge(self.plus1_node, self.plus2_node, key=self.edge_p1_p2)
        self.G.add_edge(self.plus2_node, self.a_node, key=self.edge_p2_a)
        self.G.add_edge(self.plus2_node, self.plus3_node, key=self.edge_p2_p3)
        self.G.add_edge(self.plus3_node, self.d_node, key=self.edge_p3_d)
        self.G.add_edge(self.plus3_node, self.plus4_node, key=self.edge_p3_p4)
        self.G.add_edge(self.plus4_node, self.d_node, key=self.edge_p4_d)
        self.G.add_edge(self.plus4_node, self.plus5_node, key=self.edge_p4_p5)
        self.G.add_edge(self.plus5_node, self.c_node, key=self.edge_p5_c_left)
        self.G.add_edge(self.plus5_node, self.c_node, key=self.edge_p5_c_right)

        self.G.add_edge(self.d_node, self.plus6_node, key=self.edge_d_p6)
        self.G.add_edge(self.plus6_node, self.b_node, key=self.edge_p6_b)
        self.G.add_edge(self.plus6_node, self.plus7_node, key=self.edge_p6_p7)
        self.G.add_edge(self.plus7_node, self.c_node, key=self.edge_p7_c_left)
        self.G.add_edge(self.plus7_node, self.c_node, key=self.edge_p7_c_right)

        self.G.add_edge(self.a_node, self.plus8_node, key=self.edge_a_p8)
        self.G.add_edge(self.plus8_node, self.b_node, key=self.edge_p8_b)
        self.G.add_edge(self.plus8_node, self.plus9_node, key=self.edge_p8_p9)
        self.G.add_edge(self.plus9_node, self.b_node, key=self.edge_p9_b)
        self.G.add_edge(self.plus9_node, self.c_node, key=self.edge_p9_c)

        # Test results for this graph
        self.temps = {self.a_node, self.d_node}
        self.stage_node_lists = {'e': [self.e_node,
                                       self.plus1_node,
                                       self.a_node,
                                       self.d_node,
                                       self.c_node,
                                       self.plus2_node,
                                       self.plus3_node,
                                       self.plus4_node,
                                       self.plus5_node],
                                 'd': [self.d_node,
                                       self.plus6_node,
                                       self.b_node,
                                       self.c_node,
                                       self.plus7_node],
                                 'a': [self.a_node,
                                       self.plus8_node,
                                       self.b_node,
                                       self.c_node,
                                       self.plus9_node],
                                 }
        self.stages = {name: self.G.subgraph(node_list) for name, node_list in self.stage_node_lists.items()}
        self.headless_stages = {name: self.G.subgraph(node_list[1:]) for name, node_list in self.stage_node_lists.items()}
        self.minimum_halo = np.array([4, 4, 0, 0])


@pytest.fixture(scope="module")
def gtdd():
    return GridToolsDataDepsGraph()
