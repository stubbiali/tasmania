import inspect
import networkx as nx
from gridtools.frontend.crappy.expression import Equation, ExpressionNamed, ExpressionBinaryOperator, \
												 ExpressionUnaryOperator, ExpressionConstant, ExpressionGlobal
from gridtools.intermediate_representation import graph as irgraph
from gridtools.intermediate_representation.ir import IR


class Frontend:

	def process(self, stencil_configs):
		root_equations = self._run_definitions(stencil_configs)
		graph = self._convert_equations_to_graph(root_equations)
		ir = IR(stencil_configs, graph)
		return ir

	def _run_definitions(self, stencil_configs):
		definitions_func = stencil_configs.definitions_func
		inputs_names = [key for key in stencil_configs.inputs] + [key for key in stencil_configs.constant_inputs]
		globals_names = [key for key in stencil_configs.global_inputs]

		arguments_names = inspect.getfullargspec(definitions_func)[0]
		# In the case of an object-oriented stencil, 'self' argument should be disregarded
		if arguments_names[0] == 'self':
			arguments_names = arguments_names[1:]

		arguments = dict()
		for (idx, arg) in enumerate(arguments_names):
			if arg in inputs_names:
				arguments[arg] = Equation(arg)
			elif arg in globals_names:
				arguments[arg] = ExpressionGlobal(arg)
			else:
				# Raise an error only if the argument is not a default argument
				argspec = inspect.getargspec(stencil_configs.definitions_func)
				default_values = [] if argspec[3] is None else argspec[3]
				if idx < len(arguments_names) - len(default_values):
					assert False, "Input argument {} not bound to any symbol".format(arg)

		equations = definitions_func(**arguments)
		if type(equations) is not tuple:
			equations = (equations, )
		return equations

	def _convert_equations_to_graph(self, root_equations):
		graph = nx.MultiDiGraph()
		expression_to_node_map = {}
		for root in root_equations:
			root_expression = root.expression
			assert type(root_expression) is ExpressionNamed
			self._add_expression_to_graph(graph, root_expression, expression_to_node_map)
		return graph

	def _add_expression_to_graph(self, graph, expression, expression_to_node_map):
		if expression in expression_to_node_map:
			return expression_to_node_map[expression]
		elif type(expression) is ExpressionNamed:
			return self._add_expression_named_to_graph(graph, expression, expression_to_node_map)
		elif type(expression) is ExpressionBinaryOperator:
			return self._add_expression_binary_operator_to_graph(graph, expression, expression_to_node_map)
		elif type(expression) is ExpressionUnaryOperator:
			return self._add_expression_unary_operator_to_graph(graph, expression, expression_to_node_map)
		elif type(expression) is ExpressionConstant:
			return self._add_expression_constant_to_graph(graph, expression, expression_to_node_map)
		elif type(expression) is ExpressionGlobal:
			return self._add_expression_global_to_graph(graph, expression, expression_to_node_map)

	def _add_expression_named_to_graph(self, graph, expression, expression_to_node_map):
		node = irgraph.NodeNamedExpression(name=str(expression), rank=expression.get_rank())
		if not expression.get_edges():
			graph.add_node(node)
		else:
			assert len(expression.get_edges()) == 1
			expression_edge = expression.get_edges()[0]
			expression_successor = expression_edge.get_expression_target()
			indices_offsets = expression_edge.get_indices_transformation().get_offsets_to_apply_to_source_indices()
			edge = irgraph.Edge(indices_offsets=indices_offsets)
			successor = self._add_expression_to_graph(graph, expression_successor, expression_to_node_map)
			graph.add_edge(node, successor, key=edge)
		expression_to_node_map[expression] = node
		return node

	def _add_expression_binary_operator_to_graph(self, graph, expression, expression_to_node_map):
		node = irgraph.NodeBinaryOperator(operator=str(expression))

		node_left = self._add_expression_to_graph(graph, expression.get_expression_lhs(), expression_to_node_map)
		node_right = self._add_expression_to_graph(graph, expression.get_expression_rhs(), expression_to_node_map)

		indices_offsets_left = expression.get_edge_lhs().get_indices_transformation().get_offsets_to_apply_to_source_indices()
		indices_offsets_right = expression.get_edge_rhs().get_indices_transformation().get_offsets_to_apply_to_source_indices()

		edge_left = irgraph.Edge(indices_offsets=indices_offsets_left, is_left_edge=True)
		edge_right = irgraph.Edge(indices_offsets=indices_offsets_right, is_right_edge=True)

		graph.add_edge(node, node_left, key=edge_left)
		graph.add_edge(node, node_right, key=edge_right)
		expression_to_node_map[expression] = node

		return node

	def _add_expression_unary_operator_to_graph(self, graph, expression, expression_to_node_map):
		node = irgraph.NodeUnaryOperator(operator=str(expression))
		node_out = self._add_expression_to_graph(graph, expression.get_expression_out(), expression_to_node_map)
		indices_offsets_out = expression.get_edge_out().get_indices_transformation().get_offsets_to_apply_to_source_indices()
		edge_out = irgraph.Edge(indices_offsets = indices_offsets_out)
		graph.add_edge(node, node_out, key = edge_out)
		expression_to_node_map[expression] = node
		return node

	def _add_expression_constant_to_graph(self, graph, expression, expression_to_node_map):
		assert not expression.get_subexpressions()
		node = irgraph.NodeConstant(expression.get_value())
		graph.add_node(node)
		expression_to_node_map[expression] = node
		return node

	def _add_expression_global_to_graph(self, graph, expression, expression_to_node_map):
		# Add a global node for each stage in which the underlying global variable may occur
		assert not expression.get_subexpressions()
		node = irgraph.NodeGlobal(str(expression))
		graph.add_node(node)
		expression_to_node_map[expression] = node
		return node
