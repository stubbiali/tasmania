import numbers
import inspect
import copy

from gridtools.frontend.crappy.index import Index, IndicesTransformation
from gridtools.user_interface.globals import Global


class Edge:
	def __init__(self, expression_source, expression_target, indices_transformation):
		self._expression_from = expression_source
		self._expression_to = expression_target
		self._indices_transformation = indices_transformation

	def get_expression_source(self):
		return self._expression_to

	def get_expression_target(self):
		return self._expression_to

	def set_expression_target(self, expression_target):
		self._expression_to = expression_target

	def set_indices_transformation(self, indices_transformation):
		self._indices_transformation = indices_transformation

	def get_indices_transformation(self):
		return self._indices_transformation


class Expression:
	def __init__(self):
		self._name_given_by_user = None
		self._rank = None
		self._edges = []

	def set_rank(self, rank):
		self._rank = rank

	def get_rank(self):
		return self._rank

	def add_edge(self, expression, indices_transformation=None):
		self._edges.append(Edge(expression_source      = self, 
								expression_target      = expression, 
								indices_transformation = indices_transformation))

	def get_edges(self):
		return self._edges

	def get_subexpressions(self):
		return [edge.get_expression_target() for edge in self._edges]

	def __add__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "+")

	def __radd__(self, other):
		return ExpressionBinaryOperator(_make_expression_if_necessary(other), self, "+")

	def __sub__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "-")

	def __rsub__(self, other):
		return ExpressionBinaryOperator(_make_expression_if_necessary(other), self, "-")

	def __mul__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "*")

	def __rmul__(self, other):
		return ExpressionBinaryOperator(_make_expression_if_necessary(other), self, "*")

	def __truediv__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "/")

	def __rtruediv__(self, other):
		return ExpressionBinaryOperator(_make_expression_if_necessary(other), self, "/")

	def __pow__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "**")

	def __rpow__(self, other):
		return ExpressionBinaryOperator(_make_expression_if_necessary(other), self, "**")

	def __gt__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), ">")

	def __ge__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), ">=")

	def __lt__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "<")

	def __le__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "<=")

	def __pos__(self):
		return ExpressionUnaryOperator(self, "+")

	def __neg__(self):
		return ExpressionUnaryOperator(self, "-")


class ExpressionNamed(Expression):
	def __init__(self, name_given_by_user):
		super().__init__()
		self._name_given_by_user = name_given_by_user

	def __str__(self):
		return self._name_given_by_user


class ExpressionTemporarilyHoldingIndicesInformation(Expression):
	"""
	Temporal expression type user to wrap an expression dereferenced by the user.
	The purpose of this expression type is to hold information about the indices used by the user to dereference
	the equation.
	Such information will then be moved into the edge object and this expression object will be discarded.
	"""
	def __init__(self, expression, indices):
		super().__init__()
		self.add_edge(expression)
		self._indices = indices

	def get_indices(self):
		return self._indices


class ExpressionBinaryOperator(Expression):
	def __init__(self, lhs, rhs, operator):
		super().__init__()
		self.add_edge(lhs)
		self.add_edge(rhs)
		self._lhs = lhs
		self._rhs = rhs
		self._operator = operator

	def get_edge_lhs(self):
		assert len(self.get_edges()) == 2
		return self.get_edges()[0]

	def get_edge_rhs(self):
		assert len(self.get_edges()) == 2
		return self.get_edges()[1]

	def get_expression_lhs(self):
		return self.get_edge_lhs().get_expression_target()

	def get_expression_rhs(self):
		return self.get_edge_rhs().get_expression_target()

	def __str__(self):
		return str(self._operator)


class ExpressionUnaryOperator(Expression):
	def __init__(self, expression, operator):
		super().__init__()
		self.add_edge(expression)
		self._operator = operator

	def get_edge_out(self):
		assert len(self.get_edges()) == 1
		return self.get_edges()[0]

	def get_expression_out(self):
		return self.get_edge_out().get_expression_target()

	def __str__(self):
		return str(self._operator)


class ExpressionConstant(Expression):
	def __init__(self, value):
		super().__init__()
		self._value = value

	def get_value(self):
		return self._value

	def __str__(self):
		return str(self._value)


class ExpressionGlobal(Expression):
	def __init__(self, name_given_by_user):
		super().__init__()
		self._name_given_by_user = name_given_by_user

	def __str__(self):
		return self._name_given_by_user
		

def _make_expression_if_necessary(obj):
	if isinstance(obj, Expression):
		return obj
	elif isinstance(obj, numbers.Number):
		return ExpressionConstant(obj)
	else:
		assert False, "unexpected expression type"


class Equation:
	def __init__(self, name=None):
		if name:
			self.expression = ExpressionNamed(name_given_by_user = name)
		else:
			self.expression = ExpressionNamed(name_given_by_user = self._get_name_given_by_user())

	def get_name(self):
		return self.expression._name_given_by_user

	def _get_name_given_by_user(self):
		"""
		Returns the name used to reference this object, i.e. the name of the variable created by the user.
		"""
		parent_callstack_frame = inspect.getouterframes(inspect.currentframe())[2]
		# inspect.getouterframes() returns a named tuple only since Python 3.5.
		# In order to maintain compatibility with Python <=3.4, we have to use
		# an indexing expression to retrieve the code context
		code_context = parent_callstack_frame[4]
		return code_context[0].strip().split("=")[0].strip()

	def __setitem__(self, indices, expression):
		indices = (indices,) if isinstance(indices, Index) else indices 
		self.expression.set_rank(len(indices))
		self.expression.add_edge(_make_expression_if_necessary(expression))
		self._compute_indices_transformations_dfs(self.expression, indices)
		indices_ids = self._fill_indices_transformations_with_omitted_indices(indices)
		self._set_expressions_rank_dfs(expression, len(indices_ids))

	def __getitem__(self, indices):
		return ExpressionTemporarilyHoldingIndicesInformation(self.expression, indices)

	def _compute_indices_transformations_dfs(self, expression, indices):
		edges_without_transformation = (edge for edge in expression.get_edges() if edge.get_indices_transformation() is None)
		for edge in edges_without_transformation:
			target = edge.get_expression_target()
			if type(target) is ExpressionTemporarilyHoldingIndicesInformation:
				target_indices = target.get_indices()
				indices_transformation = IndicesTransformation(source_indices=indices, target_indices=target_indices)
				edge.set_indices_transformation(indices_transformation)

				new_target = target.get_subexpressions()[0]
				edge.set_expression_target(new_target)
				self._compute_indices_transformations_dfs(new_target, target_indices)
			else:
				indices_transformation = IndicesTransformation(source_indices=indices, target_indices=indices)
				edge.set_indices_transformation(indices_transformation)
				self._compute_indices_transformations_dfs(edge.get_expression_target(), indices)

	def _fill_indices_transformations_with_omitted_indices(self, indices):
		"""
		Detect the indices that the user may have omitted when dereferencing the equations,
		and add them to the IndicesTransformation attribute of each edge.

		Parameters
		----------
		indices : tuple
			Tuple of Index's used by the user to dereference the Equation on the lhs.

		Return
		------
		set :
			Set collecting the identifiers of all the indices involved in the stage.
		"""
		indices_ids = set((index.get_id() for index in indices))
		omitted_indices_ids = copy.deepcopy(indices_ids)
		while len(omitted_indices_ids) > 0:
			indices_ids.update(omitted_indices_ids)
			omitted_indices_ids = set()
			self._fill_indices_transformations_with_omitted_indices_dfs(self.expression, indices_ids, omitted_indices_ids)
		return indices_ids

	def _fill_indices_transformations_with_omitted_indices_dfs(self, expression, indices_ids, omitted_indices_ids):
		edges = expression.get_edges()
		for edge in edges:
			indices_transformation = edge.get_indices_transformation()
			source_indices_ids = set((source_index.get_id() for source_index in indices_transformation.get_source_indices()))

			if source_indices_ids.issubset(indices_ids):
				# An equation occurring on the rhs has been dereferenced in a previous stage by omitting some indices
				missing_ids = indices_ids.difference(source_indices_ids)
				self._fill_indices_transformation_with_omitted_indices(indices_transformation, missing_ids)
				expression_target = edge.get_expression_target()
				self._fill_indices_transformations_with_omitted_indices_dfs(expression_target, indices_ids, omitted_indices_ids)
			else:
				# The equation on the lhs has been dereferenced by omitting some indices
				omitted_indices_ids.update(source_indices_ids.difference(indices_ids))

	def _fill_indices_transformation_with_omitted_indices(self, indices_transformation, missing_ids):
		source_indices_ids = set((source_index.get_id() for source_index in indices_transformation.get_source_indices()))
		for missing_id in missing_ids:
			missing_index = Index(axis=missing_id, offset=0)
			for pos, source_index_id in enumerate(source_indices_ids):
				if missing_id < source_index_id:
					indices_transformation.add_source_index(missing_index, position=pos)
					break
			else:
				# The id of the index to add is greater than any other index
				indices_transformation.add_source_index(missing_index)

	def _set_expressions_rank_dfs(self, expression, rank):
		"""
		Set the rank of each ExpressionNamed involved in the current stage.

		Parameters
		----------
		expression : obj
			An Expression.
		rank : int
			The rank.
		"""
		if type(expression) is ExpressionNamed:
			expression.set_rank(rank)
		edges = expression.get_edges()
		for edge in edges:
			expression_target = edge.get_expression_target()
			self._set_expressions_rank_dfs(expression_target, rank)
