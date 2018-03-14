"""
Classes implementing different schemes to carry out the prognostic steps of the three-dimensional 
moist isentropic dynamical core.
"""
import abc
import copy
import numpy as np

from dycore.flux_isentropic import FluxIsentropic
from dycore.horizontal_boundary import RelaxedSymmetricXZ, RelaxedSymmetricYZ
import gridtools as gt
from namelist import datatype
from storages.grid_data import GridData
from storages.state_isentropic import StateIsentropic

class PrognosticIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes to carry out the prognostic steps of 
	the three-dimensional moist isentropic dynamical core. The conservative form of the governing equations is used.

	Attributes
	----------
	ni : int
		Extent of the computational domain in the :math:`x`-direction.
	nj : int
		Extent of the computational domain in the :math:`y`-direction.
	nk : int
		Extent of the computational domain in the :math:`z`-direction.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, flux_scheme, grid, moist_on, backend):
		"""
		Constructor.

		Parameters
		----------
		flux_scheme : str 
			String specifying the flux scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj 
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.
		"""
		# Keep track of the input parameters
		self._flux_scheme, self._grid, self._moist_on, self._backend = flux_scheme, grid, moist_on, backend

		# Instantiate the class computing the numerical fluxes
		self._flux = FluxIsentropic.factory(flux_scheme, grid, moist_on)

		# Initialize the attributes representing the diagnostic step and the lateral boundary conditions
		# Remark: these should be suitably set before calling the stepping method for the first time
		self._diagnostic, self._boundary = None, None

	@property
	def diagnostic(self):
		"""
		Get the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		if self._diagnostic is None:
			raise ValueError('''The attribute which is supposed to implement the diagnostic step of the moist isentroic ''' \
							 '''dynamical core is actually :obj:`None`. Please set it correctly.''')
		return self._diagnostic

	@diagnostic.setter
	def diagnostic(self, value):
		"""
		Set the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.

		Parameter
		---------
		value : obj
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		self._diagnostic = value

	@property
	def boundary(self):
		"""
		Get the attribute implementing the horizontal boundary conditions.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			Instance of the derived class of :class:`~dycore.horizontal_boundary.HorizontalBoundary` implementing
			the horizontal boundary conditions.
		"""
		if self._boundary is None:
			raise ValueError('''The attribute which is supposed to implement the horizontal boundary conditions ''' \
							 '''is actually :obj:`None`. Please set it correctly.''')
		return self._boundary

	@boundary.setter
	def boundary(self, value):
		"""
		Set the attribute implementing the horizontal boundary conditions.

		Parameter
		---------
		value : obj
			Instance of the derived class of :class:`~dycore.horizontal_boundary.HorizontalBoundary` implementing the 
			horizontal boundary conditions.
		"""
		self._boundary = value

	@property
	def nb(self):
		"""
		Get the number of boundary layers.

		Return
		------
		int :
			The number of boundary layers.
		"""
		return self._flux.nb

	@abc.abstractmethod
	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly collecting useful diagnostics.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""

	def step_coupling_physics_with_dynamics(self, dt, state_now, state_prv, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward by coupling physics with
		dynamics, i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

		state_prv : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped taking only the horizontal derivatives into account. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

			This may be the output of :meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection`.
		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` collecting the following variables:
			
			* change_over_time_in_air_potential_temperature (unstaggered).

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""
		# Initialize the output state
		state_new = StateIsentropic(state_now.time + dt, self._grid)

		# Extract current time conservative model variables
		s_now  = state_now['isentropic_density'].values[:,:,:,0]
		U_now  = state_now['x_momentum_isentropic'].values[:,:,:,0]
		V_now  = state_now['y_momentum_isentropic'].values[:,:,:,0]
		Qv_now = state_now['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc_now = state_now['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr_now = state_now['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Extract provisional conservative model variables
		s_prv  = state_prv['isentropic_density'].values[:,:,:,0]
		U_prv  = state_prv['x_momentum_isentropic'].values[:,:,:,0]
		V_prv  = state_prv['y_momentum_isentropic'].values[:,:,:,0]
		Qv_prv = state_prv['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc_prv = state_prv['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr_prv = state_prv['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Extract the vertical velocity
		w = diagnostics['change_over_time_in_air_potential_temperature'].values[:,:,:,0]

		# The first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil_stepping_coupling_physics_with_dynamics is None:
			self._initialize_stencil_stepping_coupling_physics_with_dynamics(s_now)

		# Set stencil's inputs
		self._set_inputs_of_stencil_stepping_coupling_physics_with_dynamics(dt, w,
																			s_now, U_now, V_now, Qv_now, Qc_now, Qr_now,
																			s_prv, U_prv, V_prv, Qv_prv, Qc_prv, Qr_prv)

		# Run the stencil
		self._stencil_stepping_coupling_physics_with_dynamics.compute()

		# Set the lowest and highest layers
		self._out_s_new[:,:,:self.nb], self._out_s_new[:,:,-self.nb:] = s_prv[:,:,:self.nb], s_prv[:,:,-self.nb:]
		self._out_U_new[:,:,:self.nb], self._out_U_new[:,:,-self.nb:] = U_prv[:,:,:self.nb], U_prv[:,:,-self.nb:]
		self._out_V_new[:,:,:self.nb], self._out_V_new[:,:,-self.nb:] = V_prv[:,:,:self.nb], V_prv[:,:,-self.nb:]
		if self._moist_on:
			self._out_Qv_new[:,:,:self.nb], self._out_Qv_new[:,:,-self.nb:] = Qv_prv[:,:,:self.nb], Qv_prv[:,:,-self.nb:]
			self._out_Qc_new[:,:,:self.nb], self._out_Qc_new[:,:,-self.nb:] = Qc_prv[:,:,:self.nb], Qc_prv[:,:,-self.nb:]
			self._out_Qr_new[:,:,:self.nb], self._out_Qr_new[:,:,-self.nb:] = Qr_prv[:,:,:self.nb], Qr_prv[:,:,-self.nb:]

		# Update the output state
		state_new.add(air_isentropic_density = self._out_s_new, 
					  x_momentum_isentropic  = self._out_U_new, 
					  y_momentum_isentropic  = self._out_V_new)
		if self._moist_on:
			state_new.add(water_vapor_isentropic_density         = self._out_Qv_new, 
					  	  cloud_liquid_water_isentropic_density  = self._out_Qc_new,
					  	  precipitation_water_isentropic_density = self._out_Qr_new)

		return state_new

	@staticmethod
	def factory(time_scheme, flux_scheme, grid, moist_on, backend):
		"""
		Static method returning an instace of the derived class implementing the time stepping scheme specified 
		by :data:`time_scheme`, using the flux scheme specified by :data:`flux_scheme`.

		Parameters
		----------
		time_scheme : str
			String specifying the time stepping method to implement. Either:

			* 'forward_euler', for the forward Euler scheme;
			* 'centered', for a centered scheme.

		flux_scheme : str 
			String specifying the scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj 
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.Mode` specifying the backend for the GT4Py's stencils.

		Return
		------
		obj :
			An instace of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if time_scheme == 'forward_euler':
			return PrognosticIsentropicForwardEuler(flux_scheme, grid, moist_on, backend)
		elif time_scheme == 'centered':
			return PrognosticIsentropicCentered(flux_scheme, grid, moist_on, backend)
		else:
			raise ValueError('Unknown time integration scheme.')

	def _allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(self, s_, u_, v_):
		"""
		Allocate the attributes which serve as inputs to the GT4Py's stencils which step the solution
		disregarding the vertical advection.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Instantiate a GT4Py's Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will carry the input fields
		self._in_s_   = np.zeros_like(s_)
		self._in_u_   = np.zeros_like(u_)
		self._in_v_   = np.zeros_like(v_)
		self._in_mtg_ = np.zeros_like(s_)
		self._in_U_   = np.zeros_like(s_)
		self._in_V_   = np.zeros_like(s_)
		if self._moist_on:
			self._in_Qv_ = np.zeros_like(s_)
			self._in_Qc_ = np.zeros_like(s_)
			self._in_Qr_ = np.zeros_like(s_)

	def _allocate_outputs_of_stencils_stepping_neglecting_vertical_advection(self, s_):
		"""
		Allocate the Numpy arrays which will store the solution updated by neglecting the vertical advection.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density.
		"""
		# Allocate the Numpy arrays which will store the output fields
		# Note: allocation is performed here, i.e., the first time the entry-point method is invoked,
		# so to make this step independent of the boundary conditions type
		self._out_s_ = np.zeros_like(s_)
		self._out_U_ = np.zeros_like(s_)
		self._out_V_ = np.zeros_like(s_)
		if self._moist_on:
			self._out_Qv_ = np.zeros_like(s_)
			self._out_Qc_ = np.zeros_like(s_)
			self._out_Qr_ = np.zeros_like(s_)

	def _set_inputs_of_stencils_stepping_neglecting_vertical_advection(self, dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencils which step the solution
		disregarding the vertical advection.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
			at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
			at current time.
		p_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
		mtg_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
			at current time.
		U_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
			at current time.
		V_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
			at current time.
		Qv_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of water vapour at current time.
		Qc_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of cloud water at current time.
		Qr_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of precipitation water at current time.
		"""
		# Time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		
		# Current state
		self._in_s_[:,:,:]   = s_[:,:,:]
		self._in_u_[:,:,:]   = u_[:,:,:]
		self._in_v_[:,:,:]   = v_[:,:,:]
		self._in_mtg_[:,:,:] = mtg_[:,:,:]
		self._in_U_[:,:,:]   = U_[:,:,:]
		self._in_V_[:,:,:]   = V_[:,:,:]
		if self._moist_on:
			self._in_Qv_[:,:,:] = Qv_[:,:,:]
			self._in_Qc_[:,:,:] = Qc_[:,:,:]
			self._in_Qr_[:,:,:] = Qr_[:,:,:]

	def _initialize_stencil_stepping_coupling_physics_with_dynamics(self, s):
		"""
		Initialize the GT4Py's stencil in charge of stepping the solution by coupling physics with dynamics,
		i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` representing the isentropic density.
		"""
		# Allocate stencil's inputs
		self._allocate_inputs_of_stencil_stepping_coupling_physics_with_dynamics(s)

		# Allocate stencil's outputs
		self._allocate_outputs_of_stencil_stepping_coupling_physics_with_dynamics(s)

		# Instantiate stencil
		ni, nj, nk = s.shape[0], s.shape[1], s.shape[2] - 2 * self.nb
		_domain = gt.domain.Rectangle((0, 0, self.nb), 
									  (ni - 1, nj - 1, self.nb + nk - 1))
		_mode = self._backend

		# Instantiate the first stencil
		if not self._moist_on:
			self._stencil_stepping_coupling_physics_with_dynamics = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_coupling_physics_with_dynamics,
				inputs = {'in_w': self._in_w, 'now_s': self._in_s_now, 'prv_s': self._in_s_prv, 
						  'now_U': self._in_U_now, 'prv_U': self._in_U_prv, 
						  'now_V': self._in_V_now, 'prv_V': self._in_V_prv},
				global_inputs = {'dt': self._dt},
				outputs = {'new_s': self._out_s_new, 'new_U': self._out_U_new, 'new_V': self._out_V_new},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_coupling_physics_with_dynamics = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_coupling_physics_with_dynamics,
				inputs = {'in_w': self._in_w, 'now_s': self._in_s_now, 'prv_s': self._in_s_prv, 
						  'now_U': self._in_U_now, 'prv_U': self._in_U_prv, 
						  'now_V': self._in_V_now, 'prv_V': self._in_V_prv,
						  'now_Qv': self._in_Qv_now, 'prv_Qv': self._in_Qv_prv, 
						  'now_Qc': self._in_Qc_now, 'prv_Qc': self._in_Qc_prv, 
						  'now_Qr': self._in_Qr_now, 'prv_Qr': self._in_Qr_prv},
				global_inputs = {'dt': self._dt},
				outputs = {'new_s': self._out_s_new, 'new_U': self._out_U_new, 'new_V': self._out_V_new,
						   'new_Qv': self._out_Qv_new, 'new_Qc': self._out_Qc_new, 'new_Qr': self._out_Qr_new},	
				domain = _domain, 
				mode = _mode)

	def _allocate_inputs_of_stencil_stepping_coupling_physics_with_dynamics(self, s):
		"""
		Allocate the attributes which serve as inputs to the GT4Py's stencil which step the solution
		by coupling physics with dynamics, i.e., accounting for the change over time in potential temperature.

		Parameters
		----------
		s : array_like 
			:class:`numpy.ndarray` representing the isentropic density.
		"""
		# Instantiate a GT4Py's Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy array which will represent the vertical velocity
		self._in_w = np.zeros_like(s)

		# Allocate the Numpy arrays which will represent the current time model variables
		self._in_s_now = np.zeros_like(s)
		self._in_U_now = np.zeros_like(s)
		self._in_V_now = np.zeros_like(s)
		if self._moist_on:
			self._in_Qv_now = np.zeros_like(s)
			self._in_Qc_now = np.zeros_like(s)
			self._in_Qr_now = np.zeros_like(s)

		# Allocate the Numpy arrays which will represent the provisional model variables
		self._in_s_prv = np.zeros_like(s)
		self._in_U_prv = np.zeros_like(s)
		self._in_V_prv = np.zeros_like(s)
		if self._moist_on:
			self._in_Qv_prv = np.zeros_like(s)
			self._in_Qc_prv = np.zeros_like(s)
			self._in_Qr_prv = np.zeros_like(s)

	def _allocate_outputs_of_stencil_stepping_coupling_physics_with_dynamics(self, s):
		"""
		Allocate the Numpy arrays which will store the solution updated by coupling physics with dynamics.

		Parameters
		----------
		s : array_like 
			:class:`numpy.ndarray` representing the the isentropic density.
		"""
		self._out_s_new = np.zeros_like(s)
		self._out_U_new = np.zeros_like(s)
		self._out_V_new = np.zeros_like(s)
		if self._moist_on:
			self._out_Qv_new = np.zeros_like(s)
			self._out_Qc_new = np.zeros_like(s)
			self._out_Qr_new = np.zeros_like(s)

	def _set_inputs_of_stencil_stepping_coupling_physics_with_dynamics(self, dt, w,
																	   s_now, s_prv, U_now, U_prv, V_now, V_prv,
																	   Qv_now = None, Qv_prv = None, 
																	   Qc_now = None, Qc_prv = None, 
																	   Qr_now = None, Qr_prv = None):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencil which steps the solution
		by resolving the vertical advection, i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		w : array_like
			:class:`numpy.ndarray` representing the vertical velocity, i.e., the change over time in potential temperature.
		s_now : array_like 
			:class:`numpy.ndarray` representing the current isentropic density. 
		s_prv : array_like 
			:class:`numpy.ndarray` representing the provisional isentropic density. 
		U_now : array_like 
			:class:`numpy.ndarray` representing the current :math:`x`-momentum.
		U_prv : array_like 
			:class:`numpy.ndarray` representing the provisional :math:`x`-momentum.
		V_now : array_like 
			:class:`numpy.ndarray` representing the current :math:`y`-momentum.
		V_prv : array_like 
			:class:`numpy.ndarray` representing the provisional :math:`y`-momentum.
		Qv_now : array_like 
			:class:`numpy.ndarray` representing the current isentropic density of water vapor.
		Qv_prv : array_like 
			:class:`numpy.ndarray` representing the provisional isentropic density of water vapor.
		Qc_now : array_like 
			:class:`numpy.ndarray` representing the current isentropic density of cloud liquid water.
		Qc_prv : array_like 
			:class:`numpy.ndarray` representing the provisional isentropic density of cloud liquid water.
		Qr_now : array_like 
			:class:`numpy.ndarray` representing the current isentropic density of precipitation water.
		Qr_prv : array_like 
			:class:`numpy.ndarray` representing the provisional isentropic density of precipitation water.
		"""
		# Time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Vertical velocity
		self._in_w[:,:,:] = w[:,:,:]

		# Current state
		self._in_s_now[:,:,:] = s_now[:,:,:]
		self._in_U_now[:,:,:] = U_now[:,:,:]
		self._in_V_now[:,:,:] = V_now[:,:,:]
		if self._moist_on:
			self._in_Qv_now[:,:,:] = Qv_now[:,:,:]
			self._in_Qc_now[:,:,:] = Qc_now[:,:,:]
			self._in_Qr_now[:,:,:] = Qr_now[:,:,:]

		# Provisional state
		self._in_s_prv[:,:,:] = s_prv[:,:,:]
		self._in_U_prv[:,:,:] = U_prv[:,:,:]
		self._in_V_prv[:,:,:] = V_prv[:,:,:]
		if self._moist_on:
			self._in_Qv_prv[:,:,:] = Qv_prv[:,:,:]
			self._in_Qc_prv[:,:,:] = Qc_prv[:,:,:]
			self._in_Qr_prv[:,:,:] = Qr_prv[:,:,:]

	@abc.abstractmethod
	def defs_stencil_stepping_coupling_physics_with_dynamics(dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv,
															 Qv_now = None, Qv_prv = None, 
															 Qc_now = None, Qc_prv = None,
															 Qr_now = None, Qr_prv = None):
		"""
		GT4Py's stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		w : array_like
			:class:`numpy.ndarray` representing the vertical velocity, i.e., the change over time in potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		U_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		V_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		Qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		Qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		Qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		s_new : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		U_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		V_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		Qv_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		Qc_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		Qr_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
		"""
		

class PrognosticIsentropicForwardEuler(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement
	the forward Euler scheme carrying out the prognostic step of the three-dimensional moist isentropic dynamical core.

	Attributes
	----------
	time_levels : int
		Number of time levels the scheme relies on.
	steps : int
		Number of steps the scheme entails.
	"""
	def __init__(self, flux_scheme, grid, moist_on, backend):
		"""
		Constructor.
		
		Parameters
		----------
		flux_scheme : str 
			String specifying the flux scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT$Py's stencils.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of 
		:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend)

		# Number of time levels and steps entailed
		self.time_levels = 1
		self.steps = 1

		# The pointers to the stencils' compute function
		# They will be re-directed when the forward method is invoked for the first time
		self._stencil_stepping_neglecting_vertical_advection_first = None
		self._stencil_stepping_neglecting_vertical_advection_second = None

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward via the forward Euler method.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly collecting useful diagnostics.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""
		# Initialize the output state
		state_new = StateIsentropic(state.time + dt, self._grid)

		# Extract the model variables which are needed
		s   = state['air_isentropic_density'].values[:,:,:,0]
		u   = state['x_velocity'].values[:,:,:,0]
		v   = state['y_velocity'].values[:,:,:,0]
		U   = state['x_momentum_isentropic'].values[:,:,:,0]
		V   = state['y_momentum_isentropic'].values[:,:,:,0]
		mtg = state['montgomery_potential'].values[:,:,:,0]
		Qv	= None if not self._moist_on else state['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc	= None if not self._moist_on else state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr	= None if not self._moist_on else state['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_   = self.boundary.from_physical_to_computational_domain(s)
		u_   = self.boundary.from_physical_to_computational_domain(u)
		v_   = self.boundary.from_physical_to_computational_domain(v)
		mtg_ = self.boundary.from_physical_to_computational_domain(mtg)
		U_   = self.boundary.from_physical_to_computational_domain(U)
		V_   = self.boundary.from_physical_to_computational_domain(V)
		Qv_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv)
		Qc_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc)
		Qr_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr)

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_stepping_neglecting_vertical_advection_first is None:
			self._initialize_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs_of_stencils_stepping_neglecting_vertical_advection(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Run the compute function of the stencil stepping the isentropic density and the water constituents,
		# and providing provisional values for the momentums
		self._stencil_stepping_neglecting_vertical_advection_first.compute()

		# Bring the updated density and water constituents back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new  = self.boundary.from_computational_to_physical_domain(self._out_s_, (nx, ny, nz))
		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv_, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc_, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr_, (nx, ny, nz))

		# Apply the boundary conditions on the updated isentropic density and water constituents
		self.boundary.apply(s_new, s)
		if self._moist_on:
			self.boundary.apply(Qv_new, Qv)
			self.boundary.apply(Qc_new, Qc)
			self.boundary.apply(Qr_new, Qr)

		# Compute the provisional isentropic density; this may be scheme-dependent
		if self._flux_scheme in ['upwind', 'centered']:
			s_prov = s_new
		elif self._flux_scheme in ['maccormack']:
			s_prov = .5 * (s + s_new)

		# Diagnose the Montgomery potential from the provisional isentropic density
		state_prov = StateIsentropic(state.time + .5 * dt, self._grid, air_isentropic_density = s_prov) 
		gd = self.diagnostic.get_diagnostic_variables(state_prov, state['air_pressure'].values[0,0,0,0])

		# Extend the update isentropic density and Montgomery potential to accomodate the horizontal boundary conditions
		self._prv_s[:,:,:]   = self.boundary.from_physical_to_computational_domain(s_prov)
		self._prv_mtg[:,:,:] = self.boundary.from_physical_to_computational_domain(gd['montgomery_potential'].values[:,:,:,0])

		# Run the compute function of the stencil stepping the momentums
		self._stencil_stepping_neglecting_vertical_advection_second.compute()

		# Bring the momentums back to the original dimensions
		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz)) 

		# Apply the boundary conditions on the momentums
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)

		# Update the output state
		state_new.add(air_isentropic_density                 = s_new, 
					  x_momentum_isentropic                  = U_new, 
					  y_momentum_isentropic                  = V_new,
					  water_vapor_isentropic_density         = Qv_new, 
					  cloud_liquid_water_isentropic_density  = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		return state_new

	def _initialize_stencils_stepping_neglecting_vertical_advection(self, s_, u_, v_):
		"""
		Initialize the GT4Py's stencils implementing the forward Euler scheme.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
			at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
			at current time.
		"""
		# Allocate the Numpy arrays which will serve as inputs to the first stencil
		self._allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Allocate the Numpy arrays which will store temporary fields
		self._allocate_temporaries_of_stencils_stepping_neglecting_vertical_advection(s_)

		# Allocate the Numpy arrays which will store the output fields
		self._allocate_outputs_of_stencils_stepping_neglecting_vertical_advection(s_)

		# Set the computational domain and the backend
		ni, nj, nk = s_.shape[0] - 2 * self.nb, s_.shape[1] - 2 * self.nb, s_.shape[2]
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + ni - 1, self.nb + nj - 1, nk - 1))
		_mode = self._backend

		# Instantiate the first stencil
		if not self._moist_on:
			self._stencil_stepping_neglecting_vertical_advection_first = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_first,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._prv_U, 'out_V': self._prv_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_neglecting_vertical_advection_first = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_first,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,  
						  'in_Qv': self._in_Qv_, 'in_Qc': self._in_Qc_, 'in_Qr': self._in_Qr_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._prv_U, 'out_V': self._prv_V,
						   'out_Qv': self._out_Qv_, 'out_Qc': self._out_Qc_, 'out_Qr': self._out_Qr_},
				domain = _domain, 
				mode = _mode)

		# Instantiate the second stencil
		self._stencil_stepping_neglecting_vertical_advection_second = gt.NGStencil( 
			definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_second,
			inputs = {'in_s': self._prv_s, 'in_mtg': self._prv_mtg, 'in_U': self._prv_U, 'in_V': self._prv_V},
			global_inputs = {'dt': self._dt},
			outputs = {'out_U': self._out_U_, 'out_V': self._out_V_},
			domain = _domain, 
			mode = _mode)

	def _allocate_temporaries_of_stencils_stepping_neglecting_vertical_advection(self, s_):
		"""
		Allocate the Numpy arrays which will store temporary fields.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		"""
		self._prv_U   = np.zeros_like(s_)
		self._prv_V   = np.zeros_like(s_)
		self._prv_s   = np.zeros_like(s_)
		self._prv_mtg = np.zeros_like(s_)

	def _defs_stencil_stepping_neglecting_vertical_advection_first(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  											   in_Qv = None, in_Qc = None, in_Qr = None):
		"""
		GT4Py's stencil stepping the isentropic density and the water constituents via the forward Euler scheme.
		Further, it computes provisional values for the momentums, i.e., it updates the momentums disregarding
		the forcing terms involving the Montgomery potential.
		
		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the current time.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the current time.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the current time.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_U : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		out_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of water vapour.
		out_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of cloud water.
		out_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._moist_on:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y, \
			flux_Qv_x, flux_Qv_y, flux_Qc_x, flux_Qc_y, flux_Qr_x, flux_Qr_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr)

		out_s[i, j, k] = in_s[i, j, k] - dt * ((flux_s_x[i, j, k] - flux_s_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_s_y[i, j, k] - flux_s_y[i, j-1, k]) / self._grid.dy)
		out_U[i, j, k] = in_U[i, j, k] - dt * ((flux_U_x[i, j, k] - flux_U_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_U_y[i, j, k] - flux_U_y[i, j-1, k]) / self._grid.dy)
		out_V[i, j, k] = in_V[i, j, k] - dt * ((flux_V_x[i, j, k] - flux_V_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_V_y[i, j, k] - flux_V_y[i, j-1, k]) / self._grid.dy)
		if self._moist_on:
			out_Qv[i, j, k] = in_Qv[i, j, k] - dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy)
			out_Qc[i, j, k] = in_Qc[i, j, k] - dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy)
			out_Qr[i, j, k] = in_Qr[i, j, k] - dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def _defs_stencil_stepping_neglecting_vertical_advection_second(self, dt, in_s, in_mtg, in_U, in_V):
		"""
		GT4Py's stencil stepping the momentums via a one-time-level scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential diagnosed from the stepped isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.

		Returns
		-------
		out_U : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_U = gt.Equation()
		out_V = gt.Equation()

		# Computations
		out_U[i, j, k] = in_U[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = in_V[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy

		return out_U, out_V

	def defs_stencil_stepping_coupling_physics_with_dynamics(dt, s_now, U_now, V_now, s_prv, U_prv, V_prv,
															 Qv_now = None, Qc_now = None, Qr_now = None,
															 Qv_prv = None, Qc_prv = None, Qr_prv = None):
		"""
		GT4Py's stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		U_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		V_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		Qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		Qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		s_new : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		U_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		V_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		Qv_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		Qc_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		Qr_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		new_s = gt.Equation()
		new_U = gt.Equation()
		new_V = gt.Equation()
		if self._moist_on:
			new_Qv = gt.Equation()
			new_Qc = gt.Equation()
			new_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_z, flux_U_z, flux_V_z = self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, 
																		  now_U, prv_U, now_V, prv_V)
		else:	
			flux_s_z, flux_U_z, flux_V_z, flux_Qv_z, flux_Qc_z, flux_Qr_z = \
				self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, now_U, prv_U, now_V, prv_V,
											   now_Qv, prv_Qv, now_Qc, prv_Qc, now_Qr, prv_Qr)

		new_s[i, j, k] = prv_s[i, j, k] - dt * (flux_s_z[i, j, k] - flux_s_z[i, j, k+1]) / self._grid.dz
		new_U[i, j, k] = prv_U[i, j, k] - dt * (flux_U_z[i, j, k] - flux_U_z[i, j, k+1]) / self._grid.dz
		new_V[i, j, k] = prv_V[i, j, k] - dt * (flux_V_z[i, j, k] - flux_V_z[i, j, k+1]) / self._grid.dz
		if self._moist_on:
			new_Qv[i, j, k] = prv_Qv[i, j, k] - dt * (flux_Qv_z[i, j, k] - flux_Qv_z[i, j, k+1]) / self._grid.dz
			new_Qc[i, j, k] = prv_Qc[i, j, k] - dt * (flux_Qc_z[i, j, k] - flux_Qc_z[i, j, k+1]) / self._grid.dz
			new_Qr[i, j, k] = prv_Qr[i, j, k] - dt * (flux_Qr_z[i, j, k] - flux_Qr_z[i, j, k+1]) / self._grid.dz

		if not self._moist_on:
			return new_s, new_U, new_V
		else:
			return new_s, new_U, new_V, new_Qv, new_Qc, new_Qr


class PrognosticIsentropicCentered(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement 
	a centered time-integration scheme to carry out the prognostic step of the three-dimensional 
	moist isentropic dynamical core.

	Attributes
	----------
	ni : int
		Extent of the computational domain in the :math:`x`-direction.
	nj : int
		Extent of the computational domain in the :math:`y`-direction.
	nk : int
		Extent of the computational domain in the :math:`z`-direction.
	time_levels : int
		Number of time levels the scheme relies on.
	steps : int
		Number of steps the scheme entails.
	"""
	def __init__(self, flux_scheme, grid, moist_on, backend):
		"""
		Constructor.
		
		Parameters
		----------
		flux_scheme : str 
			String specifying the flux scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of
		:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend)

		# Number of time levels and steps entailed
		self.time_levels = 2
		self.steps = 1

		# The pointers to the stencil's compute function
		# This will be re-directed when the forward method is invoked for the first time
		self._stencil_stepping_neglecting_vertical_advection = None

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward via a centered time-integration scheme.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly collecting useful diagnostics.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""
		# Initialize the output state
		state_new = StateIsentropic(state.time + dt, self._grid)

		# Extract the needed model variables at the current time level
		s   = state['air_isentropic_density'].values[:,:,:,0]
		u   = state['x_velocity'].values[:,:,:,0]
		v   = state['y_velocity'].values[:,:,:,0]
		U   = state['x_momentum_isentropic'].values[:,:,:,0]
		V   = state['y_momentum_isentropic'].values[:,:,:,0]
		p   = state['air_pressure'].values[:,:,:,0]
		mtg = state['montgomery_potential'].values[:,:,:,0]
		Qv	= None if not self._moist_on else state['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc	= None if not self._moist_on else state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr	= None if not self._moist_on else state['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_   = self.boundary.from_physical_to_computational_domain(s)
		u_   = self.boundary.from_physical_to_computational_domain(u)
		v_   = self.boundary.from_physical_to_computational_domain(v)
		mtg_ = self.boundary.from_physical_to_computational_domain(mtg)
		U_   = self.boundary.from_physical_to_computational_domain(U)
		V_   = self.boundary.from_physical_to_computational_domain(V)
		Qv_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv)
		Qc_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc)
		Qr_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr)

		if state_old is not None:
			# Extract the needed model variables at the previous time level
			s_old  = state_old['air_isentropic_density'].values[:,:,:,0]
			U_old  = state_old['x_momentum_isentropic'].values[:,:,:,0]
			V_old  = state_old['y_momentum_isentropic'].values[:,:,:,0]
			Qv_old = None if not self._moist_on else state_old['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_old = None if not self._moist_on else state_old['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr_old = None if not self._moist_on else state_old['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Extend the arrays to accommodate the horizontal boundary conditions
			self._s_old_  = self.boundary.from_physical_to_computational_domain(s_old)
			self._U_old_  = self.boundary.from_physical_to_computational_domain(U_old)
			self._V_old_  = self.boundary.from_physical_to_computational_domain(V_old)
			self._Qv_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv_old)
			self._Qc_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc_old)
			self._Qr_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr_old)
		elif not hasattr(self, '_s_old_'):
			# Extract the needed model variables at the previous time level
			s_old  = state['air_isentropic_density'].values[:,:,:,0]
			U_old  = state['x_momentum_isentropic'].values[:,:,:,0]
			V_old  = state['y_momentum_isentropic'].values[:,:,:,0]
			Qv_old = None if not self._moist_on else state['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_old = None if not self._moist_on else state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr_old = None if not self._moist_on else state['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Extend the arrays to accommodate the horizontal boundary conditions
			self._s_old_  = self.boundary.from_physical_to_computational_domain(s_old)
			self._U_old_  = self.boundary.from_physical_to_computational_domain(U_old)
			self._V_old_  = self.boundary.from_physical_to_computational_domain(V_old)
			self._Qv_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv_old)
			self._Qc_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc_old)
			self._Qr_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr_old)

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_stepping_neglecting_vertical_advection is None:
			self._initialize_stencil_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs_of_stencils_stepping_neglecting_vertical_advection(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
			self._s_old_, self._U_old_, self._V_old_, self._Qv_old_, self._Qc_old_, self._Qr_old_)
		
		# Run the stencil's compute function
		self._stencil_stepping_neglecting_vertical_advection.compute()
		
		# Bring the updated prognostic variables back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new = self.boundary.from_computational_to_physical_domain(self._out_s_, (nx, ny, nz))

		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz)) 

		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv_, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc_, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr_, (nx, ny, nz))

		# Apply the boundary conditions
		self.boundary.apply(s_new, s)
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)
		if self._moist_on:
			self.boundary.apply(Qv_new, Qv)
			self.boundary.apply(Qc_new, Qc)
			self.boundary.apply(Qr_new, Qr)

		# Update the output state
		state_new.add(air_isentropic_density                 = s_new, 
					  x_momentum_isentropic                  = U_new, 
					  y_momentum_isentropic                  = V_new, 
					  water_vapor_isentropic_density         = Qv_new, 
					  cloud_liquid_water_isentropic_density  = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		# Keep track of the current state for the next timestep
		self._s_old_[:,:,:] = s_[:,:,:]
		self._U_old_[:,:,:] = U_[:,:,:]
		self._V_old_[:,:,:] = V_[:,:,:]
		if self._moist_on:
			self._Qv_old_[:,:,:] = Qv_[:,:,:]
			self._Qc_old_[:,:,:] = Qc_[:,:,:]
			self._Qr_old_[:,:,:] = Qr_[:,:,:]

		return state_new

	def _initialize_stencil_stepping_neglecting_vertical_advection(self, s_, u_, v_):
		"""
		Initialize the GT4Py's stencil implementing the centered time-integration scheme.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Allocate the Numpy arrays which will store the output fields
		self._allocate_outputs_of_stencils_stepping_neglecting_vertical_advection(s_)

		# Set the computational domain and the backend
		ni, nj, nk = s_.shape[0] - 2 * self.nb, s_.shape[1] - 2 * self.nb, s_.shape[2]
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + ni - 1, self.nb + nj - 1, nk - 1))
		_mode = self._backend

		# Instantiate the stencil
		if not self._moist_on:
			self._stencil_stepping_neglecting_vertical_advection = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,
						  'old_s': self._old_s_, 'old_U': self._old_U_, 'old_V': self._old_V_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._out_U_, 'out_V': self._out_V_},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_neglecting_vertical_advection = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,
						  'in_Qv': self._in_Qv_, 'in_Qc': self._in_Qc_, 'in_Qr': self._in_Qr_,
						  'old_s': self._old_s_, 'old_U': self._old_U_, 'old_V': self._old_V_,
						  'old_Qv': self._old_Qv_, 'old_Qc': self._old_Qc_, 'old_Qr': self._old_Qr_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._out_U_, 'out_V': self._out_V_,
						   'out_Qv': self._out_Qv_, 'out_Qc': self._out_Qc_, 'out_Qr': self._out_Qr_},
				domain = _domain, 
				mode = _mode)

	def _allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(self, s_, u_, v_):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py's stencil.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Instantiate a GT4Py's Global representing the timestep and the Numpy arrays
		# which will carry the solution at the current time step
		super()._allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Allocate the Numpy arrays which will carry the solution at the previous time step
		self._old_s_ = np.zeros_like(s_)
		self._old_U_ = np.zeros_like(s_)
		self._old_V_ = np.zeros_like(s_)
		if self._moist_on:
			self._old_Qv_ = np.zeros_like(s_)
			self._old_Qc_ = np.zeros_like(s_)
			self._old_Qr_ = np.zeros_like(s_)

	def _set_inputs_of_stencils_stepping_neglecting_vertical_advection(self, dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
																		s_old_, U_old_, V_old_, Qv_old_, Qc_old_, Qr_old_):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencil.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
			at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
			at current time.
		mtg_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
			at current time.
		U_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
			at current time.
		V_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
			at current time.
		Qv_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
			at current time.
		Qc_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
			at current time.
		Qr_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
			at current time.
		s_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
			at the previous time level.
		U_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
			at the previous time level.
		V_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
			at the previous time level.
		Qv_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
			at the previous time level.
		Qc_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
			at the previous time level.
		Qr_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
			at the previous time level.
		"""
		# Update the time step and the Numpy arrays carrying the current solution
		super()._set_inputs_of_stencils_stepping_neglecting_vertical_advection(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Update the Numpy arrays carrying the solution at the previous time step
		self._old_s_[:,:,:] = s_old_[:,:,:]
		self._old_U_[:,:,:] = U_old_[:,:,:]
		self._old_V_[:,:,:] = V_old_[:,:,:]
		if self._moist_on:
			self._old_Qv_[:,:,:] = Qv_old_[:,:,:]
			self._old_Qc_[:,:,:] = Qc_old_[:,:,:]
			self._old_Qr_[:,:,:] = Qr_old_[:,:,:]

	def _defs_stencil_stepping_neglecting_vertical_advection(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
															 old_s, old_U, old_V,
					  										 in_Qv = None, in_Qc = None, in_Qr = None, 
															 old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		GT4Py's stencil implementing the centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		old_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the previous time level.
		old_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the previous time level.
		old_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the previous time level.
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the current time.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the current time.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the current time.
		old_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the previous time level.
		old_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the previous time level.
		old_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the previous time level.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_U : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		out_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of water vapour.
		out_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of cloud water.
		out_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._moist_on:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y, \
			flux_Qv_x, flux_Qv_y, flux_Qc_x, flux_Qc_y, flux_Qr_x, flux_Qr_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr)

		out_s[i, j, k] = old_s[i, j, k] - 2. * dt * ((flux_s_x[i, j, k] - flux_s_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_s_y[i, j, k] - flux_s_y[i, j-1, k]) / self._grid.dy)
		out_U[i, j, k] = old_U[i, j, k] - 2. * dt * ((flux_U_x[i, j, k] - flux_U_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_U_y[i, j, k] - flux_U_y[i, j-1, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = old_V[i, j, k] - 2. * dt * ((flux_V_x[i, j, k] - flux_V_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_V_y[i, j, k] - flux_V_y[i, j-1, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy
		if self._moist_on:
			out_Qv[i, j, k] = old_Qv[i, j, k] - 2. * dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy)
			out_Qc[i, j, k] = old_Qc[i, j, k] - 2. * dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy)
			out_Qr[i, j, k] = old_Qr[i, j, k] - 2. * dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def defs_stencil_stepping_coupling_physics_with_dynamics(dt, s_now, U_now, V_now, s_prv, U_prv, V_prv,
															 Qv_now = None, Qc_now = None, Qr_now = None,
															 Qv_prv = None, Qc_prv = None, Qr_prv = None):
		"""
		GT4Py's stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		U_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		V_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		Qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		Qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		s_new : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		U_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		V_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		Qv_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		Qc_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		Qr_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		new_s = gt.Equation()
		new_U = gt.Equation()
		new_V = gt.Equation()
		if self._moist_on:
			new_Qv = gt.Equation()
			new_Qc = gt.Equation()
			new_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_z, flux_U_z, flux_V_z = self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, 
																		  now_U, prv_U, now_V, prv_V)
		else:	
			flux_s_z, flux_U_z, flux_V_z, flux_Qv_z, flux_Qc_z, flux_Qr_z = \
				self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, now_U, prv_U, now_V, prv_V,
											   now_Qv, prv_Qv, now_Qc, prv_Qc, now_Qr, prv_Qr)

		new_s[i, j, k] = prv_s[i, j, k] - 2. * dt * (flux_s_z[i, j, k] - flux_s_z[i, j, k+1]) / self._grid.dz
		new_U[i, j, k] = prv_U[i, j, k] - 2. * dt * (flux_U_z[i, j, k] - flux_U_z[i, j, k+1]) / self._grid.dz
		new_V[i, j, k] = prv_V[i, j, k] - 2. * dt * (flux_V_z[i, j, k] - flux_V_z[i, j, k+1]) / self._grid.dz
		if self._moist_on:
			new_Qv[i, j, k] = prv_Qv[i, j, k] - 2. * dt * (flux_Qv_z[i, j, k] - flux_Qv_z[i, j, k+1]) / self._grid.dz
			new_Qc[i, j, k] = prv_Qc[i, j, k] - 2. * dt * (flux_Qc_z[i, j, k] - flux_Qc_z[i, j, k+1]) / self._grid.dz
			new_Qr[i, j, k] = prv_Qr[i, j, k] - 2. * dt * (flux_Qr_z[i, j, k] - flux_Qr_z[i, j, k+1]) / self._grid.dz

		if not self._moist_on:
			return new_s, new_U, new_V
		else:
			return new_s, new_U, new_V, new_Qv, new_Qc, new_Qr
