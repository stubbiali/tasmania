import abc

class Adjustment:
	"""
	Abstract base class whose derived classes implement different ajustment schemes.

	Note
	----
	All the derived classes should be model-agnostic.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		"""
		self._grid = grid
		self._time_levels = None

	@property
	def time_levels(self):
		"""
		Get the attribute representing the number of time levels the dynamical core relies on.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		int :
			The number of time levels the dynamical core relies on.
		"""
		if self._time_levels is None:
			raise ValueError('''The attribute which is supposed to represent the number of time levels the ''' \
							 '''dynamical core relies on is actually :obj:`None`. Please set it properly.''')
		return self._time_levels

	@time_levels.setter
	def time_levels(self, value):
		"""
		Set the attribute representing the number of time levels the dynamical core relies on.

		Parameters
		----------
		value : int
			The number of time levels the dynamical core relies on.
		"""
		self._time_levels = value

	@abc.abstractmethod
	def __call__(self, dt, state):
		"""
		Entry-point method applying the parameterization scheme.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the timestep.
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.

		Return
		------
		state_new : obj
			:class:`~storages.grid_data.GridData` storing the output, adjusted state.
		diagnostics : obj
			:class:`~storages.grid_data.GridData` storing possible output diagnostics.
		"""
