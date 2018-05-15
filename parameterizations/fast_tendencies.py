import abc
import warnings

class FastTendency:
	"""
	Abstract base class whose derived classes implement different parameterization schemes 
	providing fast-varying tendencies. Here, *fast-varying* refers to those tendencies which
	should be calculated on the smallest model timestep. Hence, these classes should be ideally
	called *within* the dynamical core.

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

	@property
	def time_levels(self):
		"""
		Get the attribute representing the number of time levels the dynamical core relies on.
		
		Return
		------
		int :
			The number of time levels the dynamical core relies on.
		"""
		if self._time_levels is None:
			warn_msg = """The attribute representing the number of time levels the underlying dynamical core relies on """ \
					   """has not been previously set, so it is tacitly assumed it is 1.""" \
					   """If you want to manually set it, please use the ''time_levels'' property."""
			warnings.warn(warn_msg, RuntimeWarning)
			self._time_levels = 1
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
		tendencies : obj
			:class:`~storages.grid_data.GridData` storing the output tendencies.
		diagnostics : obj
			:class:`~storages.grid_data.GridData` storing possible output diagnostics.
		"""

class FastTendencyMicrophysics(FastTendency):
	"""
	Abstract base class whose derived classes implement different parameterization schemes providing 
	fast-varying microphysical tendencies. The derived classes also compute the following diagnostics:

	* the raindrop fall speed ([:math:`m \, s^{-1}`]).

	Note
	----
	Unless specified, none of the derived classes carries out the saturation adjustment.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, rain_evaporation_on, backend):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		"""
		super().__init__(grid)
		self._rain_evaporation_on, self._backend = rain_evaporation_on, backend

	@abc.abstractmethod
	def get_raindrop_fall_velocity(self, state):
		"""
		Get the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).
			
		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the raindrop fall velocity.
		"""

	@staticmethod
	def factory(micro_scheme, grid, rain_evaporation_on, backend, **kwargs):
		"""
		Static method returning an instance of the derived class implementing the microphysics scheme
		specified by :obj:`micro_scheme`.

		Parameters
		----------
		micro_scheme : str
			String specifying the microphysics parameterization scheme to implement. Either:

			* 'kessler_wrf', for the WRF version of the Kessler scheme.

		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		**kwargs :
			Keyword arguments to be forwarded to the derived class.
		"""
		if micro_scheme == 'kessler_wrf':
			from tasmania.parameterizations.fast_tendency_microphysics_kessler_wrf import FastTendencyMicrophysicsKesslerWRF
			return FastTendencyMicrophysicsKesslerWRF(grid, rain_evaporation_on, backend, **kwargs)
		else:
			raise ValueError('Unknown microphysics parameterization scheme.')
