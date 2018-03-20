class NodeNamedExpression:
    """
    This class implements the named expression node of the graph.
    E.g. the statement A[i, j] = B[i, j] + C[i, j] defines the expression B[i, j] + C[i, j],
    which has the name A[i, j]. A[i, j] is represented in the graph by an instance of this.
    """
    def __init__(self, name, rank):
        """
        :param name: The name of the expression.
        :param rank: The rank, i.e. number of dimensions of the expression.
        """
        self.name = name
        self.rank = rank
        self.access_extent = None

    def __str__(self):
        return self.name


class NodeBinaryOperator:
    """
    This class implements a binary operator node of the graph.
    E.g. the plus operator in the expression A[i, j] + B[i, j] is represented in the graph by an instance of this class.
    """
    def __init__(self, operator):
        """
        :param operator: A string representation of the operator.
        """
        self.operator = operator

    def __str__(self):
        return self.operator


class NodeConstant:
    """
    This class implements the constant node in the graph.
    E.g. the 1 constant in the expression A[i, j] + 1 is represented in the graph by an instance of this class.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)
		

class NodeGlobal:
	"""
	This class implementes the global node in the graph.
	A global variable is homogeneous througout the computational domain, but it may be time-dependent.
	E.g. the adaptive timestep dt in the Forward Euler statement out_u[i, j] = in_u[i, j] + dt * rhs[i, j] 
	is represented in the graph by an instance of this class.
	"""
	def __init__(self, name):
		self.name = name

	def __str__(self):
		return self.name


class Edge:
    """
    This class implements the edge of the Graph. It contains information about how the incident node is used,
    i.e. information about the indices used to access the incident node.
    """
    def __init__(self, *, indices_offsets, is_left_edge=None, is_right_edge=None):
        self.indices_offsets = indices_offsets
        self.is_left_edge = False if is_left_edge is None else is_left_edge
        self.is_right_edge = False if is_right_edge is None else is_right_edge
        assert not is_left_edge or not is_right_edge, \
            "A edge can be a left hand side edge, a right hand side edge or none of those." + \
            "However, it cannot be left edge and right edge at the same time."

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return super().__hash__()
