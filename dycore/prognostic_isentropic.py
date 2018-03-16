"""
Classes implementing different schemes to carry out the prognostic steps of the three-dimensional 
moist isentropic dynamical core.
"""
import abc
import copy
import numpy as np

import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic
from tasmania.dycore.horizontal_boundary import RelaxedSymmetricXZ, RelaxedSymmetricYZ
from tasmania.namelist import datatype
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic

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
			from tasmania.dycore.prognostic_isentropic_forward_euler import PrognosticIsentropicForwardEuler
			return PrognosticIsentropicForwardEuler(flux_scheme, grid, moist_on, backend)
		elif time_scheme == 'centered':
			from tasmania.dycore.prognostic_isentropic_centered import PrognosticIsentropicCentered
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
	def _defs_stencil_stepping_coupling_physics_with_dynamics(dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv,
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
