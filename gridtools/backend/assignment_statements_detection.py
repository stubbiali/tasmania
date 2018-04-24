import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils


class AssignmentStatementsDetection:
	def process(self, ir):
		ir.assignment_statements_graphs = find_topologically_sorted_assignment_statements(ir.graph)
		return ir


def find_topologically_sorted_assignment_statements(graph):
	"""
	Detects subgraphs corresponding to assignment statements.

	Parameters
	----------
	graph : GT4Py Intermediate Representation graph

	Returns
	-------
	list
		A list of subgraphs, each one corresponding to the Intermediate Representation
		form of a single assignment statement.
	"""
	# Stages can produce either stencil outputs (IR graph roots) or temporaries
	roots_and_temps = irutils.find_roots(graph)
	roots_and_temps.extend(find_topologically_sorted_temporaries(graph))

	subgraphs = list()

	for node in roots_and_temps:
		subgraph_nodes = start_dfs_assignment_statements(node, graph)
		subgraph = graph.subgraph(subgraph_nodes)
		subgraphs.append(subgraph)

	# The order in which the stages have been generated stems from a topological
	# sort of the IR graph, which has the same direction of a data dependency
	# graph. Thus, the order needs to be reversed to have an executable sequence
	# of stages
	subgraphs.reverse()

	return subgraphs


def start_dfs_assignment_statements(root, graph):
	stage_nodes = [root]
	if graph.successors(root):
		child = irutils.get_successor(graph, root)
		stage_nodes.extend(dfs_assignment_statements(child, graph))
	return stage_nodes


def dfs_assignment_statements(node, graph):

	if type(node) is irg.NodeNamedExpression:
		return [node]

	if type(node) is irg.NodeBinaryOperator:
		stage_nodes = [node]
		for succ in graph.successors(node):
			for n in dfs_assignment_statements(succ, graph):
				if n not in stage_nodes:
					stage_nodes.append(n)

		return stage_nodes

	if type(node) is irg.NodeUnaryOperator:
		stage_nodes = [node]
		for succ in graph.successors(node):
			for n in dfs_assignment_statements(succ, graph):
				if n not in stage_nodes:
					stage_nodes.append(n)

		return stage_nodes

	if type(node) is irg.NodeConstant:
		return [node]

	if type(node) is irg.NodeGlobal:
		return [node]


def find_topologically_sorted_temporaries(graph):
	"""
	Detects temporary data fields (in the sense of GridTools) in an Intermediate
	Representation graph

	Parameters
	----------
	graph : GT4Py Intermediate Representation graph

	Returns
	-------
	list
		A list of Intermediate Representation graph nodes. The order of the list
		depends on a topological sort of the graph.
	"""
	return [node for node in nx.topological_sort(graph) if irutils.is_temporary_named_expression(graph, node)]
