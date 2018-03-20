import numbers
import inspect

from gridtools.frontend.crappy.index import IndicesTransformation
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
		self._edges.append(Edge(expression_source=self, expression_target=expression, indices_transformation=indices_transformation))

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

	def __gt__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), ">")

	def __ge__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), ">=")

	def __lt__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "<")

	def __le__(self, other):
		return ExpressionBinaryOperator(self, _make_expression_if_necessary(other), "<=")


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
		assert len(self.get_edges()) == 2
		return self.get_edge_rhs().get_expression_target()

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
		"""
		Added by S. Ubbiali on 01/18/2018.
		"""
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
		self.expression.set_rank(len(indices))
		self.expression.add_edge(_make_expression_if_necessary(expression))
		self._compute_indices_transformations_dfs(self.expression, indices)

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
