import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils
from gridtools.user_interface.vertical_direction import VerticalDirection


def translate_stage_graph(graph, out_access_extents, vertical_direction):
	"""
	Translate a stage's graph into the corresponding Python-compliant statement. Vectorized syntax is used.

	:param graph				Stage as an instance of networkx.MultiDiGraph
	:param out_access_extents	Access extents for the output field, given in the form 
								(x_start, x_stop, y_start, y_stop, z_start, z_stop)
	:param vertical_direction	Specify the manner we should iterate over the vertical axis

	:return String containing the Python translation of the stage 
	"""
	# Find stages's output, i.e., graph's root
	root = irutils.find_roots(graph)[0]

	# Translate
	stage_src = start_dfs_translate(root, graph, out_access_extents, vertical_direction)

	return stage_src


def start_dfs_translate(root, graph, out_access_extents, vertical_direction):
	"""
	Start the translation of the stage. To scan the graph, the depth first search (DFS) method is used.
	
	:param root					Stage's output, i.e., graph's root
	:param graph				Stage as an instance of networkx.MultiDiGraph
	:param out_access_extents	Access extents for the output field
	:param vertical_direction	Specify the manner we should iterate over the vertical axis

	:return String containing the Python translation of the stage
	"""
	# Extract the edge outgoing the root and its ending node
	child = irutils.get_successor(graph, root)
	edge = irutils.get_out_edge(graph, root)

	# Translate
	expr = dfs_translate(child, edge, graph, out_access_extents, vertical_direction)
	expr = "{0}{1} = {2}".format(str(root), generate_indexing_string_for_named_expression(out_access_extents, [0.,0.,0.][:root.rank], vertical_direction), expr)
	
	return expr


def dfs_translate(node, edge, graph, out_access_extents, vertical_direction):
	"""
	Translate an edge and its ending node.

	:param node					Ending node of the edge
	:param edge					The edge
	:param graph				Stage as an instance of networkx.MultiDiGraph
	:param out_access_extents	Access extents for the output field
	:param vertical_direction	Specify the manner we should iterate over the vertical axis

	:return String containing the translation
	"""
	if type(node) is irg.NodeNamedExpression:
		# Convert access offsets in an indexing string
		access_offsets = list(edge.indices_offsets)
		indexing_string = generate_indexing_string_for_named_expression(out_access_extents, access_offsets, vertical_direction)

		return "{0}{1}".format(str(node), indexing_string)

	if type(node) is irg.NodeConstant:
		return str(node)

	if type(node) is irg.NodeGlobal:
		return "{}.value".format(str(node))

	if type(node) is irg.NodeBinaryOperator:
		left_node = irutils.get_successor_left(graph, node)
		right_node = irutils.get_successor_right(graph, node)

		# Propagating node data is a clean way to correctly process the case in
		# which multiple binary operations point to the same named expression,
		# e.g. a Laplacian operator
		left_edge = irutils.get_out_edge_left(graph, node)
		right_edge = irutils.get_out_edge_right(graph, node)

		left = dfs_translate(left_node, left_edge, graph, out_access_extents, vertical_direction)
		right = dfs_translate(right_node, right_edge, graph, out_access_extents, vertical_direction)

		return "({0}) {1} ({2})".format(left, str(node), right) 


def generate_indexing_string_for_named_expression(out_access_extents, access_offsets, vertical_direction):
	"""
	Given the access extents for the output field and the access offset, return the access extents for the current field.

	:param out_access_extents	Access extents for the output field
	:param access_offsets		Access offsets
	:param vertical_direction	Specify the manner we should iterate over the vertical axis

	:return Access extents as a string
	"""
	# Ensure the offsets list has at most three elements
	if len(access_offsets) > 3:
		raise ValueError("Numpy backend only supports array offsets in three dimensions to allow for dependences in the vertical direction.")		

	# Convert offsets in extents
	extents_strings = []
	for ind, off in enumerate(access_offsets):
		if (ind < 2) or (vertical_direction is VerticalDirection.PARALLEL):
			start = int(out_access_extents[2*ind] + off)
			stop = int(out_access_extents[2*ind+1] + off)
			extents_string = "{}:{}".format(start, stop)
		else:
			extents_string = "k"
			if off > 0: 
				extents_string += "+{}".format(off)
			elif off < 0:
				extents_string += "{}".format(off)

		extents_strings.append(extents_string)

	return "[{}]".format(",".join(extents_strings))
