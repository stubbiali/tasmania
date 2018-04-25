import copy
import numpy as np

import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic
from tasmania.dycore.horizontal_boundary_relaxed import RelaxedSymmetricXZ, RelaxedSymmetricYZ
from tasmania.dycore.prognostic_isentropic import PrognosticIsentropic
from tasmania.namelist import datatype, rho_water
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic
import tasmania.utils.utils as utils

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
	def __init__(self, flux_scheme, grid, moist_on, backend, coupling_physics_dynamics_on, 
				 sedimentation_on, sedimentation_flux_type, sedimentation_substeps):
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
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		physics_dynamics_coupling_on : bool
			:obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential temperature,
			:obj:`False` otherwise.
		sedimentation_on : bool
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		sedimentation_substeps : int
			Number of sub-timesteps to perform in order to integrate the sedimentation flux. 

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of 
		:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend, coupling_physics_dynamics_on, 
						 sedimentation_on, sedimentation_flux_type, sedimentation_substeps)

		# Number of time levels and steps entailed
		self.time_levels = 1
		self.steps = 1

		# Initialize the pointers to the compute functions of the stencils stepping the solution 
		# by neglecting vertical advection
		# These will be re-directed when the corresponding forward method is invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_advection_first = None
		self._stencil_stepping_by_neglecting_vertical_advection_second = None

		# Initialize the pointers to the compute functions of the stencils stepping the solution 
		# by integrating the sedimentation flux
		# These will be re-directed when the corresponding forward method is invoked for the first time
		self._stencil_computing_slow_tendencies = None
		self._stencil_ensuring_vertical_cfl_is_obeyed = None
		self._stencil_stepping_by_integrating_sedimentation_flux = None
		self._stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water = None

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, tendencies = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward 
		via the forward Euler method. Only horizontal derivates are considered; possible vertical 
		derivatives are disregarded.

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
			This is not actually used, yet it appears as default argument for compliancy with 
			the class hierarchy interface.
		tendencies : `obj`, optional
			:class:`~storages.grid_data.GridData` storing the following tendencies:

			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).

			Default is :obj:`None`.

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
		# The first time this method is invoked, initialize the GT4Py stencils
		if self._stencil_stepping_by_neglecting_vertical_advection_first is None:
			self._stencils_stepping_by_neglecting_vertical_advection_initialize(state, tendencies)

		# Update the attributes which serve as inputs to the first GT4Py stencil
		self._stencils_stepping_by_neglecting_vertical_advection_set_inputs(dt, state, tendencies)
		
		# Run the compute function of the stencil stepping the isentropic density and the water constituents,
		# and providing provisional values for the momentums
		self._stencil_stepping_by_neglecting_vertical_advection_first.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._mi, self._mj

		# Bring the updated density and water constituents back to the original dimensions
		out_s  = self.boundary.from_computational_to_physical_domain(self._out_s[:mi, :mj, :], (nx, ny, nz))
		out_Qv = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_Qv[:mi, :mj, :], (nx, ny, nz))
		out_Qc = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_Qc[:mi, :mj, :], (nx, ny, nz))
		out_Qr = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_Qr[:mi, :mj, :], (nx, ny, nz))

		# Apply the boundary conditions on the updated isentropic density and water constituents
		self.boundary.apply(out_s, state['air_isentropic_density'].values[:,:,:,0])
		if self._moist_on:
			self.boundary.apply(out_Qv, state['water_vapor_isentropic_density'].values[:,:,:,0])
			self.boundary.apply(out_Qc, state['cloud_liquid_water_isentropic_density'].values[:,:,:,0])
			self.boundary.apply(out_Qr, state['precipitation_water_isentropic_density'].values[:,:,:,0])

		# Compute the provisional isentropic density; this may be scheme-dependent
		if self._flux_scheme in ['upwind', 'centered']:
			s_prov = out_s
		elif self._flux_scheme in ['maccormack']:
			s_prov = .5 * (state['air_isentropic_density'].values[:,:,:,0] + out_s)

		# Diagnose the Montgomery potential from the provisional isentropic density
		time = utils.convert_datetime64_to_datetime(state['air_isentropic_density'].coords['time'].values[0])
		state_prov = StateIsentropic(time + .5 * dt, self._grid, air_isentropic_density = s_prov) 
		gd = self.diagnostic.get_diagnostic_variables(state_prov, state['air_pressure'].values[0,0,0,0])

		# Extend the update isentropic density and Montgomery potential to accomodate the horizontal boundary conditions
		self._in_s_prv[:mi, :mj, :]   = self.boundary.from_physical_to_computational_domain(s_prov)
		self._in_mtg_prv[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(
											gd['montgomery_potential'].values[:,:,:,0])

		# Run the compute function of the stencil stepping the momentums
		self._stencil_stepping_by_neglecting_vertical_advection_second.compute()

		# Bring the momentums back to the original dimensions
		if type(self.boundary) == RelaxedSymmetricXZ:
			out_U = self.boundary.from_computational_to_physical_domain(self._out_U[:mi, :mj, :], (nx, ny, nz), 
																		change_sign = False)
			out_V = self.boundary.from_computational_to_physical_domain(self._out_V[:mi, :mj, :], (nx, ny, nz), 
																		change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			out_U = self.boundary.from_computational_to_physical_domain(self._out_U[:mi, :mj, :], (nx, ny, nz), 
																		change_sign = True)
			out_V = self.boundary.from_computational_to_physical_domain(self._out_V[:mi, :mj, :], (nx, ny, nz), 
																		change_sign = False) 
		else:
			out_U = self.boundary.from_computational_to_physical_domain(self._out_U[:mi, :mj, :], (nx, ny, nz))
			out_V = self.boundary.from_computational_to_physical_domain(self._out_V[:mi, :mj, :], (nx, ny, nz)) 

		# Apply the boundary conditions on the momentums
		self.boundary.apply(out_U, state['x_momentum_isentropic'].values[:,:,:,0])
		self.boundary.apply(out_V, state['y_momentum_isentropic'].values[:,:,:,0])

		# Instantiate the output state
		state_new = StateIsentropic(time + dt, self._grid,
									air_isentropic_density				   = out_s,
					  				x_momentum_isentropic                  = out_U, 
					  				y_momentum_isentropic                  = out_V,
					  				water_vapor_isentropic_density         = out_Qv, 
					  				cloud_liquid_water_isentropic_density  = out_Qc,
					  				precipitation_water_isentropic_density = out_Qr)

		return state_new

	def step_integrating_sedimentation_flux(self, dt, state_now, state_prv, diagnostics = None):
		"""
		Method advancing the mass fraction of precipitation water by taking the sedimentation into account.
		For the sake of numerical stability, a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a timestep which may be smaller than that specified by the user.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* air_isentropic_density (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* height (:math:`z`-staggered);
			* mass_fraction_of_precipitation_water_in air (unstaggered).

		state_prv : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped without taking the sedimentation flux into account. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* mass_fraction_of_precipitation_water_in air (unstaggered).

			This may be the output of either
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection` or
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_coupling_physics_with_dynamics`.
		diagnostics : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` collecting the following diagnostics:

			* accumulated_precipitation (unstaggered, two-dimensional).

		Returns
		-------
		state_new : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the following updated variables:
			
			* mass_fraction_of_precipitation_water_in air (unstaggered);
			* precipitation_water_isentropic_density (unstaggered).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting the output diagnostics, i.e.:

			* accumulated_precipitation (unstaggered, two-dimensional);
			* precipitation (unstaggered, two-dimensional).
		"""
		# The first time this method is invoked, initialize the underlying GT4Py stencils
		if self._stencil_stepping_by_integrating_sedimentation_flux is None:
			self._stencils_stepping_by_integrating_sedimentation_flux_initialize()

		# Compute the smaller timestep
		dts = dt / float(self._sedimentation_substeps)

		# Update the attributes which serve as inputs to the GT4Py stencils
		nx, ny = self._grid.nx, self._grid.ny
		self._dt.value  = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		self._dts.value = 1.e-6 * dts.microseconds if dts.seconds == 0. else dts.seconds
		self._in_rho[:,:,:]       = state_now['air_density'].values[:,:,:,0]
		self._in_s[:nx,:ny,:]     = state_now['air_isentropic_density'].values[:,:,:,0]
		self._in_s_prv[:nx,:ny,:] = state_prv['air_isentropic_density'].values[:,:,:,0]
		self._in_h[:,:,:]         = state_now['height'].values[:,:,:,0]
		self._in_qr[:,:,:] 	   	  = state_now['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0] 
		self._in_qr_prv[:,:,:] 	  = state_prv['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Set the output fields
		nb = self._flux_sedimentation.nb
		self._out_s[:nx,:ny,:nb] = state_prv['air_isentropic_density'].values[:,:,:nb,0]
		self._out_s[:nx,:ny,nb:] = state_now['air_isentropic_density'].values[:,:,nb:,0]
		self._out_qr[:,:,:nb]    = state_prv['mass_fraction_of_precipitation_water_in_air'].values[:,:,:nb,0]
		self._out_qr[:,:,nb:]    = state_now['mass_fraction_of_precipitation_water_in_air'].values[:,:,nb:,0]

		# Initialize the output state
		time_now = utils.convert_datetime64_to_datetime(state_now['air_density'].coords['time'].values[0])
		state_new = StateIsentropic(time_now + dt, self._grid,
									air_density                                 = self._in_rho,
									air_isentropic_density						= self._out_s[:nx,:ny,:],
									height                                      = self._in_h,
									mass_fraction_of_precipitation_water_in_air = self._out_qr)

		# Initialize the arrays storing the precipitation and the accumulated precipitation
		precipitation = np.zeros((nx, ny, 1), dtype = datatype)
		accumulated_precipitation = np.zeros((nx, ny, 1), dtype = datatype)
		if diagnostics is not None and diagnostics['accumulated_precipitation'] is not None:
			accumulated_precipitation[:,:,:] = diagnostics['accumulated_precipitation'].values[:,:,:,0]

		# Initialize the output diagnostics
		diagnostics_out = GridData(time_now + dt, self._grid,
								   precipitation             = precipitation,
								   accumulated_precipitation = accumulated_precipitation)

		# Compute the slow tendencies over the large timestep
		self._stencil_computing_slow_tendencies.compute()

		# Perform the time-splitting procedure
		for n in range(self._sedimentation_substeps):
			# Compute the raindrop fall velocity
			self._in_vt[:,:,:] = self.microphysics.get_raindrop_fall_velocity(state_new)

			# Make sure the vertical CFL is obeyed
			self._stencil_ensuring_vertical_cfl_is_obeyed.compute()
			self._in_vt[:,:,:] = self._out_vt[:,:,:]

			# Compute the precipitation and the accumulated precipitation
			ppt = self._in_rho[:,:,-1:] * self._in_qr[:,:,-1:] * self._in_vt[:,:,-1:] * self._dts.value / rho_water
			precipitation[:,:,:] = ppt[:,:,:] / self._dts.value * 3.6e6
			accumulated_precipitation[:,:,:] += ppt[:,:,:] * 1.e3

			# Perform a small timestep
			self._stencil_stepping_by_integrating_sedimentation_flux.compute()

			# Diagnose the geometric height and the air density
			state_new.update(self.diagnostic.get_height(state_new, pt = state_now['air_pressure'].values[0,0,0,0]))
			state_new.update(self.diagnostic.get_air_density(state_new))

			# Advance the solution
			self._in_s[:,:,:]  = self._out_s[:,:,:]
			self._in_qr[:,:,:] = self._out_qr[:,:,:]

		# Diagnose the isentropic density of precipitation water
		self._stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water.compute()
		state_new.add_variables(time_now + dt,
								precipitation_water_isentropic_density = self._out_Qr[:nx,:ny,:])

		return state_new, diagnostics_out

	def _stencils_stepping_by_neglecting_vertical_advection_initialize(self, state, tendencies):
		"""
		Initialize the GT4Py stencils implementing the forward Euler scheme to step the solution 
		by neglecting vertical advection.

		Parameters
		----------
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered).
		"""
		# Deduce the shape of any (unstaggered) input field
		s = self.boundary.from_physical_to_computational_domain(state['air_isentropic_density'].values[:,:,:,0])
		mi, mj, nz = s.shape

		# Allocate the Numpy arrays which will serve as inputs to the first stencil
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(mi, mj, tendencies)

		# Allocate the Numpy arrays which will store provisional (i.e., temporary) fields
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_temporaries(mi, mj)

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_outputs(mi, mj)

		# Set the stencils' computational domain and the backend
		nb = self.nb
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0), 
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the first stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
				   'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V}
		_outputs = {'out_s': self._out_s, 'out_U': self._tmp_U, 'out_V': self._tmp_V}
		if self._moist_on:
			_inputs.update({'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc,	'in_Qr': self._in_Qr})
			_outputs.update({'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr})
		if tendencies is not None:
			if tendencies['tendency_of_mass_fraction_of_water_vapor_in_air'] is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'] is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies['tendency_of_mass_fraction_of_precipitation_water_in_air'] is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the first stencil
		self._stencil_stepping_by_neglecting_vertical_advection_first = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_first_defs,
			inputs = _inputs,
			global_inputs = {'dt': self._dt},
			outputs = _outputs,
			domain = _domain, 
			mode = _mode)

		# Instantiate the second stencil
		self._stencil_stepping_by_neglecting_vertical_advection_second = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_second_defs,
			inputs = {'in_s': self._in_s_prv, 'in_mtg': self._in_mtg_prv, 
					  'in_U': self._tmp_U, 'in_V': self._tmp_V},
			global_inputs = {'dt': self._dt},
			outputs = {'out_U': self._out_U, 'out_V': self._out_V},
			domain = _domain, 
			mode = _mode)

	def _stencils_stepping_by_neglecting_vertical_advection_allocate_temporaries(self, mi, mj):
		"""
		Allocate the Numpy arrays which will store temporary fields to be shared between the stencils
		stepping the solution by neglecting vertical advection.

		Parameters
		----------
		mi : int
			:math:`x`-extent of an input array representing an :math:`x`-unstaggered field.
		mj : int
			:math:`y`-extent of an input array representing a :math:`y`-unstaggered field.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Determine the size of the arrays
		# Even if these arrays will not be shared with the stencil in charge of coupling physics with dynamics,
		# they should be treated as they were
		li = mi if not self._physics_dynamics_coupling_on else max(mi, nx)
		lj = mj if not self._physics_dynamics_coupling_on else max(mj, ny)

		# Allocate the arrays
		self._tmp_U      = np.zeros((li, lj, nz), dtype = datatype)
		self._tmp_V      = np.zeros((li, lj, nz), dtype = datatype)
		self._in_s_prv   = np.zeros((li, lj, nz), dtype = datatype)
		self._in_mtg_prv = np.zeros((li, lj, nz), dtype = datatype)

	def _stencil_stepping_by_neglecting_vertical_advection_first_defs(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  											   	  in_Qv = None, in_Qc = None, in_Qr = None,
																	  in_qv_tnd = None, in_qc_tnd = None, in_qr_tnd = None):
		"""
		GT4Py stencil stepping the isentropic density and the water constituents via the forward Euler scheme.
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
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of the mass fraction of precipitation water.

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
		# Declare indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._moist_on:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y,  \
			flux_U_x, flux_U_y,  \
			flux_V_x, flux_V_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x,  flux_s_y,   \
			flux_U_x,  flux_U_y,   \
			flux_V_x,  flux_V_y,   \
			flux_Qv_x, flux_Qv_y,  \
			flux_Qc_x, flux_Qc_y,  \
			flux_Qr_x, flux_Qr_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
												 in_Qv, in_Qc, in_Qr, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s[i, j, k] = in_s[i, j, k] - dt * ((flux_s_x[i, j, k] - flux_s_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_s_y[i, j, k] - flux_s_y[i, j-1, k]) / self._grid.dy)

		# Advance the x-momentum
		out_U[i, j, k] = in_U[i, j, k] - dt * ((flux_U_x[i, j, k] - flux_U_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_U_y[i, j, k] - flux_U_y[i, j-1, k]) / self._grid.dy)

		# Advance the y-momentum
		out_V[i, j, k] = in_V[i, j, k] - dt * ((flux_V_x[i, j, k] - flux_V_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_V_y[i, j, k] - flux_V_y[i, j-1, k]) / self._grid.dy)
		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_Qv[i, j, k] = in_Qv[i, j, k] - \
								  dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy)
			else:
				out_Qv[i, j, k] = in_Qv[i, j, k] - \
								  dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy -
										in_s[i, j, k] * in_qv_tnd[i, j, k])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_Qc[i, j, k] = in_Qc[i, j, k] - \
								  dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy)
			else:
				out_Qc[i, j, k] = in_Qc[i, j, k] - \
								  dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy -
										in_s[i, j, k] * in_qc_tnd[i, j, k])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_Qr[i, j, k] = in_Qr[i, j, k] - \
								  dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy)
			else:
				out_Qr[i, j, k] = in_Qr[i, j, k] - \
								  dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 				(flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy -
										in_s[i, j, k] * in_qr_tnd[i, j, k])

		if not self._moist_on:
			return out_s, out_U, out_V
		else:
			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr

	def _stencil_stepping_by_neglecting_vertical_advection_second_defs(self, dt, in_s, in_mtg, in_U, in_V):
		"""
		GT4Py stencil stepping the momentums via a one-time-level scheme.

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

	def _stencil_stepping_by_coupling_physics_with_dynamics_defs(dt, in_w, 
																 in_s, in_s_prv, 
																 in_U, in_U_prv, 
																 in_V, in_V_prv,
															  	 in_Qv = None, in_Qv_prv = None, 
																 in_Qc = None, in_Qc_prv = None, 
																 in_Qr = None, in_Qr_prv = None):
		"""
		GT4Py stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		in_w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in
			potential temperature. 
		in_s : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		in_s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		in_U : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		in_U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		in_V : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		in_V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		in_Qv : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		in_Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		in_Qc : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		in_Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		in_Qr : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		in_Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		out_U : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		out_V : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		out_Qv : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		out_Qc : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		out_Qr : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
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
			flux_s_z, flux_U_z, flux_V_z = self._flux.get_vertical_fluxes(i, j, k, dt, in_w, in_s, in_s_prv, 
																		  in_U, in_U_prv, in_V, in_V_prv)
		else:	
			flux_s_z, flux_U_z, flux_V_z, flux_Qv_z, flux_Qc_z, flux_Qr_z = \
				self._flux.get_vertical_fluxes(i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv,
											   in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv)

		out_s[i, j, k] = in_s_prv[i, j, k] - dt * (flux_s_z[i, j, k] - flux_s_z[i, j, k+1]) / self._grid.dz
		out_U[i, j, k] = in_U_prv[i, j, k] - dt * (flux_U_z[i, j, k] - flux_U_z[i, j, k+1]) / self._grid.dz
		out_V[i, j, k] = in_V_prv[i, j, k] - dt * (flux_V_z[i, j, k] - flux_V_z[i, j, k+1]) / self._grid.dz
		if self._moist_on:
			out_Qv[i, j, k] = in_Qv_prv[i, j, k] - dt * (flux_Qv_z[i, j, k] - flux_Qv_z[i, j, k+1]) / self._grid.dz
			out_Qc[i, j, k] = in_Qc_prv[i, j, k] - dt * (flux_Qc_z[i, j, k] - flux_Qc_z[i, j, k+1]) / self._grid.dz
			out_Qr[i, j, k] = in_Qr_prv[i, j, k] - dt * (flux_Qr_z[i, j, k] - flux_Qr_z[i, j, k+1]) / self._grid.dz

		if not self._moist_on:
			return out_s, out_U, out_V
		else:
			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr

	def _stencils_stepping_by_integrating_sedimentation_flux_initialize(self):
		"""
		Initialize the GT4Py stencils in charge of stepping the mass fraction of precipitation water by 
		integrating the sedimentation flux.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._flux_sedimentation.nb

		# Allocate the GT4Py Globals which represent the large and small timestep
		self._dt = gt.Global()
		self._dts = gt.Global()
		
		# Allocate the Numpy arrays which will serve as stencils' inputs
		self._in_rho    = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_h      = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_qr     = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_qr_prv = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_vt     = np.zeros((nx, ny, nz  ), dtype = datatype)

		# Allocate the Numpy arrays which will be shared among different stencils
		self._tmp_s_tnd  = np.zeros((nx, ny, nz), dtype = datatype)
		self._tmp_qr_tnd = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencils' outputs
		self._out_vt = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the GT4Py stencil in charge of computing the slow tendencies
		self._stencil_computing_slow_tendencies = gt.NGStencil(
			definitions_func = self._stencil_computing_slow_tendencies_defs,
			inputs           = {'in_s': self._in_s, 'in_s_prv': self._in_s_prv,
								'in_qr': self._in_qr, 'in_qr_prv': self._in_qr_prv},
			global_inputs    = {'dt': self._dt},
			outputs          = {'out_s_tnd': self._tmp_s_tnd, 'out_qr_tnd': self._tmp_qr_tnd},
			domain           = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode             = self._backend)

		# Initialize the GT4Py stencil ensuring that the vertical CFL condition is fulfilled
		self._stencil_ensuring_vertical_cfl_is_obeyed = gt.NGStencil(
			definitions_func = self._stencil_ensuring_vertical_cfl_is_obeyed_defs,
			inputs           = {'in_h': self._in_h, 'in_vt': self._in_vt},
			global_inputs    = {'dts': self._dts},
			outputs          = {'out_vt': self._out_vt},
			domain           = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode             = self._backend)

		# Initialize the GT4Py stencil in charge of actually stepping the solution by integrating the sedimentation flux
		self._stencil_stepping_by_integrating_sedimentation_flux = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_integrating_sedimentation_flux_defs,
			inputs           = {'in_rho': self._in_rho, 'in_s': self._in_s,	'in_h': self._in_h, 
								'in_qr': self._in_qr, 'in_vt': self._in_vt, 
								'in_s_tnd': self._tmp_s_tnd, 'in_qr_tnd': self._tmp_qr_tnd},
			global_inputs    = {'dts': self._dts},
			outputs          = {'out_s': self._out_s[:nx,:ny,:], 'out_qr': self._out_qr},
			domain           = gt.domain.Rectangle((0, 0, nb), (nx - 1, ny - 1, nz - 1)),
			mode             = self._backend)

		# Initialize the GT4Py stencil clipping the negative values for the mass fraction of precipitation water,
		# and diagnosing the isentropic density of precipitation water
		self._stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water = gt.NGStencil(
			definitions_func = self._stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water_defs,
			inputs           = {'in_s': self._out_s, 'in_qr': self._out_qr},
			outputs          = {'out_qr': self._out_qr, 'out_Qr': self._out_Qr[:nx,:ny,:]},
			domain           = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode             = self._backend)

	def _stencil_computing_slow_tendencies_defs(self, dt, in_s, in_s_prv, in_qr, in_qr_prv):
		"""
		GT4Py stencil computing the slow tendencies required to resolve rain sedimentation.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the large timestep.
		in_s : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density.
		in_qr : obj
			:class:`gridtools.Equation` representing the current mass fraction of precipitation water.
		in_qr_prv : obj
			:class:`gridtools.Equation` representing the provisional mass fraction of precipitation water.

		Return
		------
		out_s_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency for the isentropic density.
		out_qr_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency for the mass fraction of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s_tnd  = gt.Equation()
		out_qr_tnd = gt.Equation()

		# Computations
		out_s_tnd[i, j, k]  = (in_s_prv[i, j, k] - in_s[i, j, k]) / dt
		out_qr_tnd[i, j, k] = (in_qr_prv[i, j, k] - in_qr[i, j, k]) / dt

		return out_s_tnd, out_qr_tnd

	def _stencil_ensuring_vertical_cfl_is_obeyed_defs(self, dts, in_h, in_vt):
		"""
		GT4Py stencil ensuring that the vertical CFL condition is fulfilled.
		This is achieved by clipping the raindrop fall velocity field: if a cell does not satisfy the CFL constraint, 
		the vertical velocity at that cell is reduced so that the local CFL number equals 0.95.

		Parameters
		----------
		dts : obj
			:class:`gridtools.Global` representing the large timestep.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the clipped raindrop fall velocity.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output fields
		tmp_cfl = gt.Equation()
		out_vt  = gt.Equation()

		# Computations
		tmp_cfl[i, j, k] = in_vt[i, j, k] * dts / (in_h[i, j, k] - in_h[i, j, k+1])
		out_vt[i, j, k]  = (tmp_cfl[i, j, k] < 0.95) * in_vt[i, j, k] + \
						   (tmp_cfl[i, j, k] > 0.95) * 0.95 * (in_h[i, j, k] - in_h[i, j, k+1]) / dts

		return out_vt

	def _stencil_stepping_by_integrating_sedimentation_flux_defs(self, dts, in_rho, in_s, in_h, in_qr, 
				  											  	 in_vt, in_s_tnd, in_qr_tnd):
		"""
		GT4Py stencil stepping the isentropic density and the mass fraction of precipitation water 
		by integrating the precipitation flux.

		Parameters
		----------
		dts : obj
			:class:`gridtools.Global` representing the small timestep.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_s : obj
			:class:`gridtools.Equation` representing the air isentropic density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the input mass fraction of precipitation water.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		in_s_tnd : obj
			:class:`gridtools.Equation` representing the contribution from the slow tendencies for the isentropic density.
		in_qr_tnd : obj
			:class:`gridtools.Equation` representing the contribution from the slow tendencies for the mass fraction of
			precipitation water.

		Return
		------
		out_s : obj
			:class:`gridtools.Equation` representing the output isentropic density.
		out_qr : obj
			:class:`gridtools.Equation` representing the output mass fraction of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output fields
		tmp_qr_st = gt.Equation()
		tmp_qr    = gt.Equation()
		out_s     = gt.Equation()
		out_qr    = gt.Equation()

		# Update isentropic density
		out_s[i, j, k] = in_s[i, j, k] + dts * in_s_tnd[i, j, k]

		# Update mass fraction of precipitation water
		tmp_dfdz = self._flux_sedimentation.get_vertical_derivative_of_sedimentation_flux(i, j, k, in_rho, in_h, in_qr, in_vt)
		tmp_qr_st[i, j, k] = in_qr[i, j, k] + dts * in_qr_tnd[i, j, k]
		tmp_qr[i, j, k] = tmp_qr_st[i, j, k] + dts * tmp_dfdz[i, j, k] / in_rho[i, j, k]
		out_qr[i, j, k] = (tmp_qr[i, j, k] > 0.) * tmp_qr[i, j, k] + (tmp_qr[i, j, k] < 0.) * tmp_qr_st[i, j, k]

		return out_s, out_qr

	def _stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water_defs(self, in_s, in_qr):
		"""
		GT4Py stencil clipping the negative values for the mass fraction of precipitation water, 
		and diagnosing the isentropic density of precipitation water.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the air isentropic density.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water in air.

		Return
		------
		out_qr : obj
			:class:`gridtools.Equation` representing the clipped mass fraction of precipitation water.
		out_Qr : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_qr = gt.Equation()
		out_Qr = gt.Equation()

		# Computations
		out_qr[i, j, k] = (in_qr[i, j, k] > 0.) * in_qr[i, j, k]
		out_Qr[i, j, k] = in_s[i, j, k] * out_qr[i, j, k]

		return out_qr, out_Qr
