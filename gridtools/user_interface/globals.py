import numbers
import inspect


class Global:
	"""
	This class should be used to represent scalar variables homogeneous throughout the computational domain
	but possibly time-dependent, as, e.g., the timestep in a time-adaptive PDE solver. 
	"""
	def __init__(self, value = 0., name = None):
		"""
		:param value	Global value
		:param name		Name given by the user; if not specified, the name is deduced from the context
		"""
		self.value = value
		if name is not None:
			self._name = name
		else:
			self._name = self._get_name_given_by_user()
	
	
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

	
	def __add__(self, other):
		"""
		Addition operator.
		"""
		if isinstance(other, Global):
			return Global(self.value + other.value)
		else:
			return Global(self.value + other)


	def __radd__(self, other):
		"""
		Reverse addition operator.
		"""
		return Global(other + self.value)


	def __iadd__(self, other):
		"""
		Addition assignment operator.
		"""
		if isinstance(other, Global):
			self.value += other.value
		else:
			self.value += other
		return self

		
	def __sub__(self, other):
		"""
		Subtraction operator.
		"""
		if isinstance(other, Global):
			return Global(self.value - other.value)
		else:
			return Global(self.value - other)


	def __rsub__(self, other):
		"""
		Reverse subtraction operator.
		"""
		return Global(other - self.value)
		

	def __isub__(self, other):
		"""
		Subtraction assignment operator.
		"""
		if isinstance(other, Global):
			self.value -= other.value
		else:
			self.value -= other
		return self


	def __mul__(self, other):
		"""
		Multiplication operator.
		"""
		if isinstance(other, Global):
			return Global(self.value * other.value)
		else:
			return Global(self.value * other)


	def __rmul__(self, other):
		"""
		Reverse multiplication operator.
		"""
		return Global(other * self.value)


	def __imul__(self, other):
		"""
		Multiplication assignment operator.
		"""
		if isinstance(other, Global):
			self.value *= other.value
		else:
			self.value *= other
		return self


	def __truediv__(self, other):
		"""
		Division operator.
		"""
		if isinstance(other, Global):
			return Global(self.value / other.value)
		else:
			return Global(self.value / other)


	def __rtruediv__(self, other):
		"""
		Reverse division operator.
		"""
		return Global(other / self.value)


	def __itruediv__(self, other):
		"""
		Division assignment operator.
		"""
		if isinstance(other, Global):
			self.value /= other.value
		else:
			self.value /= other
		return self


	def __lt__(self, other):
		"""
		Less than operator.
		"""
		if isinstance(other, Global):
			return self.value < other.value
		else:
			return self.value < other


	def __le__(self, other):
		"""
		Less than or equal to operator.
		"""
		if isinstance(other, Global):
			return self.value <= other.value
		else:
			return self.value <= other


	def __gt__(self, other):
		"""
		Greater than operator.
		"""
		if isinstance(other, Global):
			return self.value > other.value
		else:
			return self.value > other


	def __ge__(self, other):
		"""
		Greater than or equal to operator.
		"""
		if isinstance(other, Global):
			return self.value >= other.value
		else:
			return self.value >= other


	def __str__(self):
		"""
		String operator.
		"""
		return self._name
