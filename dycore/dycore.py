import abc
import numpy as np
from sympl import TimeStepper

class DynamicalCore(TimeStepper):
	"""
	Abstract base class whose derived classes implement different dynamical cores.
	The class inherits :class:`sympl.TimeStepper`.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid, self._moist_on = grid, moist_on

		# Initialize pointer to the object in charge of calculating the raindrop fall velocity
		self._microphysics = None

		# Initialize the list of parameterizations calculating fast-varying tendencies
		self._fast_tendency_params = []

	@property
	@abc.abstractmethod
	def time_levels(self):
		"""
		Get the number of time leves the dynamical core relies on.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Return
		------
		int :
			The number of time levels needed by the dynamical core.
		"""

	@property
	@abc.abstractmethod
	def microphysics(self):
		"""
		Get the attribute in charge of calculating the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Return
		------
		obj :
			Instance of a derived class of either 
			:class:`~tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics`,
			:class:`~tasmania.parameterizations.fast_tendencies.FastTendencyMicrophysics`,
			or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` in 
			charge of calculating the raindrop fall velocity.
		"""

	@microphysics.setter
	@abc.abstractmethod
	def microphysics(self, micro):
		"""
		Set the attribute in charge of calculating the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		micro : obj
			Instance of a derived class of either 
			:class:`~tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics`,
			:class:`~tasmania.parameterizations.fast_tendencies.FastTendencyMicrophysics`,
			or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` in 
			charge of calculating the raindrop fall velocity.
		"""

	@property
	@abc.abstractmethod
	def fast_tendency_parameterizations(self):
		"""
		Get the list of parameterizations calculating fast-varying tendencies.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Return
		------
		list :
			List containing instances of derived classes of 
			:class:`~tasmania.parameterizations.fast_tendencies.FastTendency` which are in charge of
			calculating fast-varying tendencies.
		"""

	@fast_tendency_parameterizations.setter
	@abc.abstractmethod
	def fast_tendency_parameterizations(self, fast_tendency_params):
		"""
		Set the list of parameterizations calculating fast-varying tendencies.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		fast_tendency_paras : list
			List containing instances of derived classes of 
			:class:`~tasmania.parameterizations.fast_tendencies.FastTendency` which are in charge of
			calculating fast-varying tendencies.
		"""

	@abc.abstractmethod
	def __call__(self, dt, state, tendencies = None, diagnostics = None):
		"""
		Call operator advancing the input state one step forward. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` object representing the time step.
		state : obj 
			The current state, as an instance of :class:`~tasmania.storages.grid_data.GridData` or one of its derived classes.
		tendencies : `obj`, optional 
			:class:`~tasmania.storages.grid_data.GridData` storing tendencies. Default is :obj:`None`.
		diagnostics : `obj`, optional 
			:class:`~tasmania.storages.grid_data.GridData` storing diagnostics. Default is :obj:`None`.

		Return
		------
		state_new : obj
			The state at the next time level. This is of the same class of :data:`state`.
		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` storing output diagnostics.
		"""

	@abc.abstractmethod
	def get_initial_state(self, *args):
		"""
		Get the initial state.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		*args :
			The arguments depend on the specific dynamical core which the derived class implements.

		Return
		------
		obj :
			The initial state, as an instance of :class:`~tasmania.storages.grid_data.GridData` or one of its derived classes.
		"""

	def update_topography(self, time):
		"""
		Update the underlying (time-dependent) topography.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self._grid.update_topography(time)

	@staticmethod
	def factory(model, *args, **kwargs):
		"""
		Static method returning an instance of the derived class implementing the dynamical core specified by :obj:`model`.

		Parameters
		----------
		model : str
			String specifying the dynamical core to implement. Either:

				* 'isentropic_conservative', for the isentropic dynamical core based on the conservative form of \
					the governing equations;
				* 'isentropic_nonconservative', for the isentropic dynamical core based on the nonconservative form of \
					the governing equations.

		*args :
			Positional arguments to forward to the derived class.
		**kwargs :
			Keyword arguments to forward to the derived class.
			
		Return
		------
		obj :
			Instance of the derived class implementing the specified model.
		"""
		if model == 'isentropic_conservative':
			from tasmania.dycore.dycore_isentropic import DynamicalCoreIsentropic
			return DynamicalCoreIsentropic(*args, **kwargs)
		elif model == 'isentropic_nonconservative':
			from tasmania.dycore.dycore_isentropic_nonconservative import DynamicalCoreIsentropicNonconservative
			return DynamicalCoreIsentropicNonconservative(*args, **kwargs)
