import unittest
import graphviz as gv

from gridtools.frontend.crappy import index as idx
from gridtools.frontend.crappy import expression as expr


class TestDAGGeneration(unittest.TestCase):
    def test_dag_generations(self):
        i = idx.Index()
        j = idx.Index()

        A = expr.Equation()
        B = expr.Equation()
        C = expr.Equation()
        D = expr.Equation()

        B[i, j] = A[i+10, j+11] + 10
        C[i, j] = A[i+1, j+2] + B[j+1, i+1]
        D[i, j] = C[i, j] / 2 + 11

        self._print_access_pattern_dag(D.expression, "access_pattern_dag")

    def _print_access_pattern_dag(self, expression, filename):
        dag = self._make_access_pattern_dag(expression)
        dag.render(filename)

    def _make_access_pattern_dag(self, expression):
        dag = gv.Digraph(format="svg")
        indices_names = [bytes([b"i"[0]+i]).decode("ascii") for i in range(0, expression.get_rank())]
        indices = [idx.Index(name=idx_name) for idx_name in indices_names]
        self._make_access_pattern_dag_dfs(indices, expression, dag)
        return dag

    def _make_access_pattern_dag_dfs(self, indices, expression, dag):
        expression_id = str(id(expression))
        expression_label = str(expression)
        dag.node(expression_id, label=expression_label)

        for edge in expression.get_edges():
            transformed_indices = edge.get_indices_transformation().transform_indices(indices)

            target = edge.get_expression_target()
            target_id = str(id(target))
            target_label = str(target)
            edge_label = self._make_edge_label_with_access_pattern_information(edge, transformed_indices)

            dag.node(target_id, target_label)
            dag.edge(target_id, expression_id, label=edge_label)

            self._make_access_pattern_dag_dfs(transformed_indices, target, dag)

    def _make_edge_label_with_access_pattern_information(self, edge, indices):
        target = edge.get_expression_target()
        if type(target) is expr.ExpressionNamed:
            labels = [self._make_index_label(idx) for idx in indices]
            return "[" + ",".join(labels) + "]"
        else:
            return ""

    def _make_index_label(self, index):
        label = index.get_name()
        if index.get_offset() > 0:
            label = label + "+" + str(index.get_offset())
        elif index.get_offset() < 0:
            label = label + "-" + str(index.get_offset())
        return label

if __name__ == '__main__':
    unittest.main()
