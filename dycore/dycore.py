import abc
import numpy as np
from sympl import TimeStepper

class DynamicalCore(TimeStepper):
	"""
	Abstract base class whose derived classes implement different dynamical cores.
	The class inherits :class:`sympl.TimeStepper`.

	Attributes
	----------
	microphysics : obj
		Derived class of :class:`~parameterizations.microphysics.Microphysics` taking care of the cloud microphysics.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid, self._moist_on = grid, moist_on

		# Initialize pointer to the object taking care of the microphysics
		self._microphysics = None

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
	def microphysics(self):
		"""
		Get the attribute in charge of computing the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Return
		------
		obj :
			Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
			or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` which provides the raindrop fall velocity.
		"""

	@microphysics.setter
	@abc.abstractmethod
	def microphysics(self, micro):
		"""
		Set the attribute in charge of computing the raindrop fall velocity.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		micro : obj
			Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
			or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` which provides the raindrop fall velocity.
		"""

	@abc.abstractmethod
	def __call__(self, dt, state, diagnostics = None, tendencies = None):
		"""
		Call operator advancing the input state one step forward. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` object representing the time step.
		state : obj 
			The current state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.
		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing diagnostics. Default is :obj:`None`.
		tendencies : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing tendencies. Default is :obj:`None`.

		Return
		------
		state_new : obj
			The state at the next time level. This is of the same class of :data:`state`.
		diagnostics_out : obj
			:class:`~storages.grid_data.GridData` storing output diagnostics.
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
			The initial state, as an instance of :class:`~storages.grid_data.GridData` or one of its derived classes.
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

			* 'isentropic', for the hydrostatic, isentropic dynamical core;
			* 'isentropic_isothermal', for the hydrostatic, isentropic, isothermal dynamical core.

		*args :
			Positional arguments to forward to the derived class.
		**kwargs :
			Keyword arguments to forward to the derived class.
			
		Return
		------
		obj :
			Instance of the derived class implementing the specified model.
		"""
		if model == 'isentropic':
			from tasmania.dycore.dycore_isentropic import DynamicalCoreIsentropic
			return DynamicalCoreIsentropic(*args, **kwargs)
