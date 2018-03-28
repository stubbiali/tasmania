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
		self.microphysics = None

	@property
	def time_levels(self):
		"""
		Get the number of time leves the dynamical core relies on.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Return
		------
		int :
			The number of time levels needed by the dynamical core.
		"""
		return self

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
			:class:`~storages.grid_data.GridData` possibly storing diagnostics. Default is :obj:`None`.
		tendencies : `obj`, optional 
			:class:`~storages.grid_data.GridData` possibly storing tendencies. Default is :obj:`None`.

		Return
		------
		obj :
			The state at the next time level. This is of the same class of :data:`state`.
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
