"""
This module contains utility function to operate on the intermediate representation.
"""
import os
import tempfile

import graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import gridtools.intermediate_representation.graph as irg


def get_successor(g, node):
	successors = g.successors(node)
	if not successors:
		raise LookupError("Graph node {} doesn't have successors".format(str(node)))
	if len(successors) != 1:
		raise LookupError("Graph node {} has more than one successor".format(str(node)))
	return successors[0]


def get_successor_left(g, node):
	edges = g.out_edges(node, keys=True)
	for _, successor, key in edges:
		if key.is_left_edge:
			return successor
	raise LookupError("Graph node {} doesn't have a left out edge".format(str(node)))


def get_successor_right(g, node):
	edges = g.out_edges(node, keys=True)
	for _, successor, key in edges:
		if key.is_right_edge:
			return successor
	raise LookupError("Graph node {} doesn't have a right out edge".format(str(node)))


def get_out_edge(g, node):
	edges = g.out_edges(node, keys=True)
	if not edges:
		raise LookupError("Graph node {} doesn't have an out edge".format(str(node)))
	if len(edges) != 1:
		raise LookupError("Graph node {} has more than one out edge".format(str(node)))
	_, _, key = edges[0]
	return key


def get_out_edge_left(g, node):
	edges = g.out_edges(node, keys=True)
	for _, _, key in edges:
		if key.is_left_edge:
			return key
	raise LookupError("Graph node {} doesn't have a left out edge".format(str(node)))


def get_out_edge_right(g, node):
	edges = g.out_edges(node, keys=True)
	for _, _, key in edges:
		if key.is_right_edge:
			return key
	raise LookupError("Graph node {} doesn't have a right out edge".format(str(node)))


def is_temporary_named_expression(g, node):
	return g.predecessors(node) and g.successors(node) and (type(node) is irg.NodeNamedExpression)


def find_roots(g):
	roots = [node for node in g if not g.predecessors(node)]
	return roots


def find_leaves(g):
	leaves = [node for node in g if not g.successors(node)]
	return leaves


def convert_to_graphviz(nx_graph, format='svg', engine='dot'):
	"""
	Convert a NetworkX directed graph to a graphviz object.

	Parameters
	----------
	nx_graph : NetworkX graph
		A directed graph created with NetworkX
	format : str
		Format of the output when saving the graph to file. Default is SVG.
		Please consult http://www.graphviz.org/doc/info/output.html for a list
		of available formats.
	engine : str
		graphviz program to use in order to generate node layout.
		Accepted values: 'dot', 'neato', 'twopi', 'circo', 'fdp', 'sfdp',
		'patchwork', 'osage'
	"""
	gv_graph = graphviz.Digraph(name=nx_graph.name, format=format, engine=engine)

	# Default graph attributes
	# Use these to change the default appearance of graph, nodes, and edges
	gv_graph.graph_attr.update(nx_graph.graph.get('graph', {}))
	gv_graph.node_attr.update(nx_graph.graph.get('node', {}))
	gv_graph.edge_attr.update(nx_graph.graph.get('edge', {}))

	# Add nodes
	for n, nodedata in nx_graph.nodes(data=True):
		str_nodedata = dict((k, str(v)) for k, v in nodedata.items())
		gv_graph.node(str(id(n)), label=str(n), **str_nodedata)

	# Loop over edges
	for u, v, key, edgedata in nx_graph.edges(data=True, keys=True):
		str_edgedata = dict((k, str(v)) for k, v in edgedata.items() if k != 'key')
		gv_graph.edge(str(id(u)), str(id(v)), key=str(key), **str_edgedata)

	return gv_graph


def save_graph(nx_graph, path, format='svg', engine='dot', view=False):
	"""
	Write an Intermediate Representation graph to file using graphviz.

	Please note that this function will write *both* an image file (with the
	image format suffix) and a DOT file with a name exactly corresponding to
	the 'path' argument (no suffixes)

	Parameters
	----------
	nx_graph : NetworkX graph
		A graph created with NetworkX
	path : str
		Location where to save to output file.
	format : str
		Format of the output file. Default is SVG.
		Please consult http://www.graphviz.org/doc/info/output.html for a list
		of available formats.
	engine : str
		graphviz program to use in order to generate node layout.
		Accepted values: 'dot', 'neato', 'twopi', 'circo', 'fdp', 'sfdp',
		'patchwork', 'osage'
	view : boolean
		If True, the resulting file will automatically be opened  with
		your system’s default viewer application for the file type.
	"""
	gv_graph = convert_to_graphviz(nx_graph, format=format, engine=engine)
	gv_graph.render(path, view=view)


def plot_graph(nx_graph, engine='dot'):
	"""
	Plot an Intermediate Representation graph using matplotlib and graphviz.

	Please notice that graphviz graphs can be visualized directly inside a
	Jupyter notebook or qtconsole just by printing them.
	Use gridtools.intermediate_representation.convert_to_graphviz() to obtain
	a printable object.

	This function is meant to be used when zooming/panning is needed, or
	when running in environments without the rich output capabilities of
	notebooks and qtconsoles.

	Parameters
	----------
	nx_graph : NetworkX graph
		A graph created with NetworkX
	engine : str
		graphviz program to use in order to generate node layout.
		Accepted values: 'dot', 'neato', 'twopi', 'circo', 'fdp', 'sfdp',
		'patchwork', 'osage'
	"""
	# Use a temporary PNG file to store the plot image. The file will be closed
	# and deleted upon exiting the context manager
	with tempfile.NamedTemporaryFile(suffix='.png') as fp:
		save_graph(nx_graph, path=fp.name[:-4], format='png', engine=engine)
		img = mpimg.imread(fp.name)
		imgplot = plt.imshow(img)
		plt.show()
		# Remove the DOT file
		os.remove(fp.name[:-4])


def assert_graphs_are_equal(a_graph, b_graph):
	a_roots = {str(node): node for node in a_graph.nodes() if not a_graph.predecessors(node)}
	b_roots = {str(node): node for node in b_graph.nodes() if not b_graph.predecessors(node)}

	for a_root_name, a_root in a_roots.items():
		assert a_root_name in b_roots
		b_root = b_roots.pop(a_root_name)
		assert type(a_root) is irg.NodeNamedExpression
		assert type(b_root) is irg.NodeNamedExpression
		_assert_graphs_are_equal_dfs(a_graph, a_root, b_graph, b_root, a_to_b_map={}, b_to_a_map={})

	assert not b_roots


def _assert_graphs_are_equal_dfs(a_graph, a_node, b_graph, b_node, a_to_b_map, b_to_a_map):
	assert type(a_node) is type(b_node)
	assert str(a_node) == str(b_node)

	is_a_node_already_visited = a_node in a_to_b_map
	is_b_node_already_visited = b_node in b_to_a_map
	assert is_a_node_already_visited == is_b_node_already_visited
	if is_a_node_already_visited:
		assert b_node is a_to_b_map[a_node]
		assert a_node is b_to_a_map[b_node]
		return
	else:
		a_to_b_map[a_node] = b_node
		b_to_a_map[b_node] = a_node

	if type(a_node) is irg.NodeBinaryOperator:
		assert len(a_graph.out_edges(a_node)) == 2
		assert len(b_graph.out_edges(b_node)) == 2

		a_left_edge = get_out_edge_left(a_graph, a_node)
		b_left_edge = get_out_edge_left(b_graph, b_node)
		assert a_left_edge == b_left_edge

		a_right_edge = get_out_edge_right(a_graph, a_node)
		b_right_edge = get_out_edge_right(b_graph, b_node)
		assert a_right_edge == b_right_edge

		a_left_successor = get_successor_left(a_graph, a_node)
		b_left_successor = get_successor_left(b_graph, b_node)
		_assert_graphs_are_equal_dfs(a_graph, a_left_successor, b_graph, b_left_successor, a_to_b_map, b_to_a_map)

		a_right_successor = get_successor_right(a_graph, a_node)
		b_right_successor = get_successor_right(b_graph, b_node)
		_assert_graphs_are_equal_dfs(a_graph, a_right_successor, b_graph, b_right_successor, a_to_b_map, b_to_a_map)

	else:
		a_has_successors = bool(a_graph.successors(a_node))
		b_has_successors = bool(b_graph.successors(b_node))
		assert a_has_successors == b_has_successors

		if a_has_successors:
			a_edge = get_out_edge(a_graph, a_node)
			b_edge = get_out_edge(b_graph, b_node)
			assert a_edge == b_edge

			a_successor = get_successor(a_graph, a_node)
			b_successor = get_successor(b_graph, b_node)
			_assert_graphs_are_equal_dfs(a_graph, a_successor, b_graph, b_successor, a_to_b_map, b_to_a_map)
