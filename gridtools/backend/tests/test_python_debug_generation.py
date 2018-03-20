import numpy as np
import networkx as nx

import gridtools as gt
from gridtools.user_interface.ngstencil import NGStencilConfigs
from gridtools.intermediate_representation.ir import IR
from gridtools.intermediate_representation import graph as irgraph
from gridtools.backend.compute_extents import AccessExtentsComputation
from gridtools.backend.python_debug_generation import PythonDebugGeneration


def test_python_code_rendering():
    ir = _create_intermediate_representation()
    expected_python_code = ["import traitlets",
                            "import numpy as np",
                            "try:",
                            "    import ipdb",
                            "except traitlets.config.configurable.MultipleInstanceError:",
                            "    # We are likely inside a Jupyter notebook",
                            "    from IPython.core.debugger import Tracer",
                            "def dummy_stencil_debug(B, A):",
                            "    try: ipdb.set_trace()",
                            "    except NameError: Tracer()()  # Cavalry's here!",
                            "    for i in range(1, 2):",
                            "        for j in range(0, 2):",
                            "            B[i,j] = A[i-1,j+1]"]
    actual_python_code = _render_python_code_without_blank_lines(ir)
    assert actual_python_code == expected_python_code


def _test_python_debug_generation():
    ir = _create_intermediate_representation()
    ir = PythonDebugGeneration().process(ir)
    ir.computation_func()

    actual_b = ir.stencil_configs.outputs["B"]
    expected_b = np.array([[0, 0, 0, 0],
                          [1, 2, 3, 0],
                          [11, 12, 13, 0]], dtype=np.float)
    expected_b = expected_b[:, :]
    assert np.array_equal(actual_b, expected_b)


def _create_intermediate_representation():
    """
    Create an intermediate representation for the following definitions:
    B[i, j] = A[i-1, j+1]
    """
    def dummy_stencil():
        pass
    a = np.array([[0, 1, 2, 3],
                  [10, 11, 12, 13],
                  [20, 21, 22, 23]], dtype=np.float)
    a = a[:, :, np.newaxis]
    b = np.zeros_like(a)
    domain = gt.domain.Rectangle((1, 0), (2, 2))
    stencil_configs = NGStencilConfigs(definitions_func=dummy_stencil,
                                       inputs={"A": a},
                                       outputs={"B": b},
                                       domain=domain,
                                       mode=None)

    a = irgraph.NodeNamedExpression("A", 2)
    b = irgraph.NodeNamedExpression("B", 2)
    edge_ba = irgraph.Edge(indices_offsets=[-1, 1])

    graph = nx.MultiDiGraph()
    graph.add_edge(b, a, key=edge_ba)

    ir = IR(stencil_configs, graph)
    ir = AccessExtentsComputation().process(ir)
    ir.assignment_statements_graphs = [ir.graph.copy()]
    return ir


def _render_python_code_without_blank_lines(ir):
    python_code = PythonDebugGeneration()._render_python_code(ir)
    lines = python_code.split("\n")
    lines_without_blank_lines = [line for line in lines if line.strip() != ""]
    return lines_without_blank_lines
