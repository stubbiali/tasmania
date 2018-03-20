import numpy as np
import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils


class AccessExtentsComputation:
	def process(self, ir):
		self._compute_access_extents(ir.graph)
		ir.minimum_halo = self._compute_minimum_halo(ir.graph)
		return ir

	def _compute_access_extents(self, graph):
		"""
		Computes access extents for NamedExpression nodes in an Intermediate
		Representation graph.

		The access extent information is saved into the nodes
		themselves, so the graph is modified in-place.

		Parameters
		----------
		graph : GT4Py Intermediate Representation graph
		"""
		for root in irutils.find_roots(graph):
			# Stencil outputs have zero-valued access extent
			root_extent = np.zeros(2*root.rank, dtype=np.int)
			root.access_extent = root_extent

			child = irutils.get_successor(graph, root)
			edge = irutils.get_out_edge(graph, root)
			edge_offset = self._convert_offsets_to_extent_format(edge.indices_offsets)
			new_extent = edge_offset + root_extent
			self._dfs_access_extents(child, graph, new_extent)

	def _compute_minimum_halo(self, graph):
		"""
		Computes the minimum halo of a stencil given its Intermediate Representation
		graph. The minimum halo is obtained by computing the minimum enclosing extent
		among all the extents of the graph's leaf nodes (i.e. the stencil's inputs)

		Parameters
		----------
		graph : GT4Py Intermediate Representation graph

		Returns
		-------
		halo : NumPy array of 4 positive integers
			The minimum halo of the stencil. Has the same format of the access
			extents. For details, see
			AccessExtentsComputation._convert_offsets_to_extent_format()
		"""
		roots = irutils.find_roots(graph)
		halo = np.zeros(2*roots[0].rank, dtype=np.int)
		for leaf in irutils.find_leaves(graph):
			if type(leaf) is irg.NodeNamedExpression:
				assert leaf.access_extent is not None, \
					("Leaf node is expected to contain access extent information. "
					 "Hint: did you compute the graph's access extents before "
					 "calling this function?")
				halo = np.maximum(halo, leaf.access_extent)
		return halo

	def _dfs_access_extents(self, node, graph, previous_extent):

		if type(node) is irg.NodeNamedExpression:
			if node.access_extent is None:
				node.access_extent = previous_extent
			else:
				node.access_extent = np.maximum(node.access_extent, previous_extent)
			try:
				child = irutils.get_successor(graph, node)
				edge = irutils.get_out_edge(graph, node)
				edge_offset = self._convert_offsets_to_extent_format(edge.indices_offsets)
				new_extent = edge_offset + previous_extent
				self._dfs_access_extents(child, graph, new_extent)

			except LookupError:
				# This node is a graph leaf
				assert not graph.successors(node), \
					("Internal error: graph's named expression nodes are supposed "
					 "to have either zero or one successors")
				return

		if type(node) is irg.NodeConstant:
			return

		if type(node) is irg.NodeGlobal:
			return

		if type(node) is irg.NodeBinaryOperator:
			for _, successor_node, edge in graph.out_edges(node, keys=True):
				edge_offset = self._convert_offsets_to_extent_format(edge.indices_offsets)
				new_extent = edge_offset + previous_extent
				self._dfs_access_extents(successor_node, graph, new_extent)

	def _convert_offsets_to_extent_format(self, offsets):
		"""
		The access extent format is:
			[Offset in negative X direction,
			 Offset in positive X direction,
			 Offset in negative Y direction,
			 Offset in positive Y direction,
			 Offset in negative Z direction,
			 Offset in positive Z direction]
		"""
		extent = np.zeros(2*len(offsets), dtype=np.int)

		for ind, off in enumerate(offsets):
			if off < 0:
				extent[2*ind] = -off
			elif off > 0:
				extent[2*ind+1] = off

		return extent
