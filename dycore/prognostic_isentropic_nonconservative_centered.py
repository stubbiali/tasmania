import copy
import numpy as np

import gridtools as gt
from tasmania.dycore.horizontal_boundary_relaxed import RelaxedSymmetricXZ, RelaxedSymmetricYZ
from tasmania.dycore.prognostic_isentropic_nonconservative import PrognosticIsentropicNonconservative
from tasmania.namelist import datatype, rho_water
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic
import tasmania.utils.utils as utils

class PrognosticIsentropicNonconservativeCentered(PrognosticIsentropicNonconservative):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative` 
	to implement a centered time-integration scheme to carry out the prognostic step of the three-dimensional 
	moist isentropic dynamical core. The nonconservative form of the governing equations, expressed using isentropic
	coordinates, is used.

	Attributes
	----------
	time_levels : int
		Number of time levels the scheme relies on.
	steps : int
		Number of steps the scheme entails.
	"""
	def __init__(self, flux_scheme, grid, moist_on, backend, coupling_physics_dynamics_on, sedimentation_on):
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
			:obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential 
			temperature, :obj:`False` otherwise.
		sedimentation_on : bool
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative.factory` of
		:class:`~dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend, coupling_physics_dynamics_on, sedimentation_on)

		# Number of time levels and steps entailed
		self.time_levels = 2
		self.steps = 1

		# Initialize the pointer to the compute functions of the stencils stepping the solution 
		# by neglecting vertical advection; these will be re-directed when the corresponding forward method 
		# is invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_advection_unstaggered = None
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_x  = None
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_y  = None

		# Initialize the pointers to the compute functions of the stencils stepping the solution 
		# by integrating the sedimentation flux; these will be re-directed when the corresponding forward method 
		# is invoked for the first time
		self._stencil_stepping_by_integrating_sedimentation_flux = None
		self._stencil_computing_slow_tendency = None

		# Boolean flag to quickly assess whether we are within the first time step
		self._is_first_timestep = True

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None, tendencies = None):
		"""
		Method advancing the prognostic model variables one time step forward via a centered time-integration 
		scheme. Only horizontal derivates are considered; possible vertical derivatives are disregarded.

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
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly storing diagnostics.
			For the time being, this is not actually used.
		tendencies : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly storing tendencies.
			For the time being, this is not actually used.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
		"""
		# Initialize the output state
		time = utils.convert_datetime64_to_datetime(state['air_isentropic_density'].coords['time'].values[0])
		state_new = StateIsentropic(time + dt, self._grid)

		# The first time this method is invoked, initialize the GT4Py stencils
		if self._stencil_stepping_by_neglecting_vertical_advection_unstaggered is None:
			self._stencils_stepping_by_neglecting_vertical_advection_initialize(state)

		# Update the attributes which serve as inputs to the GT4Py stencils
		self._stencils_stepping_by_neglecting_vertical_advection_set_inputs(dt, state, state_old)
		
		# Run the stencils' compute functions
		self._stencil_stepping_by_neglecting_vertical_advection_unstaggered.compute()
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_x.compute()
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_y.compute()
		
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._mi, self._mj

		# Bring the updated isentropic density back to the original dimensions
		s_new = self.boundary.from_computational_to_physical_domain(self._out_s[:mi, :mj, :], (nx, ny, nz))

		# Bring the updated velocity components back to the original dimensions
		if type(self.boundary) == RelaxedSymmetricXZ:
			u_new = self.boundary.from_computational_to_physical_domain(self._out_u[:mi+1, :mj, :], (nx+1,   ny, nz), 
																		change_sign = False)
			v_new = self.boundary.from_computational_to_physical_domain(self._out_v[:mi, :mj+1, :], (  nx, ny+1, nz), 
																		change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			u_new = self.boundary.from_computational_to_physical_domain(self._out_u[:mi+1, :mj, :], (nx+1,   ny, nz), 
																		change_sign = True)
			v_new = self.boundary.from_computational_to_physical_domain(self._out_v[:mi, :mj+1, :], (  nx, ny+1, nz), 
																		change_sign = False) 
		else:
			u_new = self.boundary.from_computational_to_physical_domain(self._out_u[:mi+1, :mj, :], (nx+1,   ny, nz))
			v_new = self.boundary.from_computational_to_physical_domain(self._out_v[:mi, :mj+1, :], (  nx, ny+1, nz)) 

		# Bring the updated water constituents back to the original dimensions
		qv_new = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_qv[:mi, :mj, :], (nx, ny, nz))
		qc_new = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_qc[:mi, :mj, :], (nx, ny, nz))
		qr_new = None if not self._moist_on else \
				 self.boundary.from_computational_to_physical_domain(self._out_qr[:mi, :mj, :], (nx, ny, nz))

		# Apply the boundary conditions
		self.boundary.apply(s_new, state['air_isentropic_density'].values[:,:,:,0])
		self.boundary.apply(u_new, state['x_velocity'].values[:,:,:,0])
		self.boundary.apply(v_new, state['y_velocity'].values[:,:,:,0])
		if self._moist_on:
			self.boundary.apply(qv_new, state['mass_fraction_of_water_vapor_in_air'].values[:,:,:,0])
			self.boundary.apply(qc_new, state['mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0])
			self.boundary.apply(qr_new, state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0])

		# Update the output state
		state_new.add_variables(time + dt,
					  			air_isentropic_density                      = s_new, 
					  			x_velocity                                  = u_new, 
					  			y_velocity                                  = v_new, 
					  			mass_fraction_of_water_vapor_in_air         = qv_new, 
					  			mass_fraction_of_cloud_liquid_water_in_air  = qc_new,
					  			mass_fraction_of_precipitation_water_in_air = qr_new)

		# Keep track of the current state for the next timestep
		self._in_s_old[  :mi,   :mj, :] = self._in_s[  :mi,   :mj, :]
		self._in_u_old[:mi+1,   :mj, :] = self._in_u[:mi+1,   :mj, :]
		self._in_v_old[  :mi, :mj+1, :] = self._in_v[  :mi, :mj+1, :]
		if self._moist_on:
			self._in_qv_old[:mi, :mj, :] = self._in_qv[:mi, :mj, :]
			self._in_qc_old[:mi, :mj, :] = self._in_qc[:mi, :mj, :]
			self._in_qr_old[:mi, :mj, :] = self._in_qr[:mi, :mj, :]

		# At this point, the first timestep is surely over
		self._is_first_timestep = False

		return state_new

	def step_integrating_sedimentation_flux(self, dt, state_now, state_prv, diagnostics = None):
		"""
		Method advancing the mass fraction of precipitation water by taking the sedimentation into account.
		For the sake of numerical stability, a time-splitting strategy is pursued, i.e., the sedimentation flux
		is integrated using a timestep which may be smaller than that specified by the user.

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

			* air_density (unstaggered);
			* air_isentropic_density (unstaggered);
			* air_pressure (:math:`z`-staggered);
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
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting the output diagnostics, i.e.:

			* accumulated_precipitation (unstaggered, two-dimensional);
			* precipitation (unstaggered, two-dimensional).
		"""
		# The first time this method is invoked, initialize the underlying GT4Py stencils
		if self._stencil_stepping_by_integrating_sedimentation_flux is None:
			self._stencils_stepping_by_integrating_sedimentation_flux_initialize()

		# Compute the number of substeps required to obey the vertical CFL condition, 
		# so retaining numerical stability
		h = state_now['height'].values[:,:,:,0]
		vt = self.microphysics.get_raindrop_fall_velocity(state_now)
		cfl = np.max(vt[:,:,:] * (1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds) / \
					 (h[:,:,:-1] - h[:,:,1:]))
		substeps = max(np.ceil(cfl), 1.)
		dts = dt / substeps

		# Update the attributes which serve as inputs to the GT4Py stencils
		nx, ny = self._grid.nx, self._grid.ny
		self._dt.value  = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		self._dts.value = 1.e-6 * dts.microseconds if dts.seconds == 0. else dts.seconds
		self._in_s[:nx, :ny, :]     = self._in_s_old[:nx, :ny, :]
		self._in_s_prv[:nx, :ny, :] = state_prv['air_isentropic_density'].values[:, :, :, 0]
		self._in_qr[:, :, :] 	   	= self._in_Qr_old[:nx, :ny, :] / self._in_s_old[:nx, :ny, :] 
		self._in_qr_prv[:, :, :] 	= state_prv['mass_fraction_of_precipitation_water_in_air'].values[:, :, :, 0]

		# Set the output fields
		nb = self._flux_sedimentation.nb
		self._out_s[:nx, :ny, :nb] = state_prv['air_isentropic_density'].values[:, :, :nb, 0]
		self._out_s[:nx, :ny, nb:] = self._in_s[:nx, :ny, nb:]
		self._out_qr[:, :, :nb]    = state_prv['mass_fraction_of_precipitation_water_in_air'].values[:, :, :nb, 0]
		self._out_qr[:, :, nb:]    = self._in_qr[:nx, :ny, nb:]

		# Initialize the output state
		time_now = utils.convert_datetime64_to_datetime(state_now['air_density'].coords['time'].values[0])
		state_new = StateIsentropic(time_now + dt, self._grid,
									air_density									= self._in_rho,
									air_isentropic_density						= self._out_s[:nx, :ny, :],
									height										= self._in_h,
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

		# Compute the slow tendencies from the large timestepping
		self._stencil_computing_slow_tendencies.compute()

		# Advance the solution
		self._in_s[:nx, :ny, :nb] = state_now['air_isentropic_density'].values[:, :, :nb, 0]
		self._in_s[:nx, :ny, nb:] = self._out_s[:nx, :ny, nb:]
		self._in_qr[:, :, :nb]    = state_now['mass_fraction_of_precipitation_water_in_air'].values[:, :, :nb, 0]
		self._in_qr[:, :, nb:]    = self._out_qr[:, :, nb:]

		# Perform the time-splitting procedure
		for _ in range(int(substeps)):
			# Diagnose the geometric height and the air density
			state_new.update(self.diagnostic.get_height(state_new, pt = state_now['air_pressure'].values[0,0,0,0]))
			state_new.update(self.diagnostic.get_air_density(state_new))

			# Compute the raindrop fall velocity
			self._in_vt[:,:,:] = self.microphysics.get_raindrop_fall_velocity(state_new)

			# Compute the precipitation and the accumulated precipitation
			ppt = self._in_rho[:,:,-1:] * self._in_qr[:,:,-1:] * self._in_vt[:,:,-1:] * self._dts.value / rho_water
			precipitation[:,:,:] = ppt[:,:,:] / self._dts.value * 3.6e6
			accumulated_precipitation[:,:,:] += ppt[:,:,:] * 1.e3

			# Perform a small timestep
			self._stencil_stepping_by_integrating_sedimentation_flux.compute()

			# Advance the solution
			self._in_s[:, :, nb:]  = self._out_s[:, :, nb:]
			self._in_qr[:, :, nb:] = self._out_qr[:, :, nb:]

		# Pop out useless variables from output state
		state_new.pop('air_density')
		state_new.pop('air_isentropic_density')
		state_new.pop('height')

		# Diagnose the isentropic density of precipitation water
		self._stencil_clipping_mass_fraction_and_diagnosing_isentropic_density_of_precipitation_water.compute()
		state_new.add_variables(time_now + dt, 
								mass_fraction_of_precipitation_water_in_air = self._out_Qr[:nx, :ny, :])

		return state_new, diagnostics_out

	def _stencils_stepping_by_neglecting_vertical_advection_initialize(self, state):
		"""
		Initialize the GT4Py stencils implementing a time-integration centered scheme to step the solution 
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

		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(mi, mj)

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_outputs(mi, mj)

		# Set inputs and outputs for the stencil stepping the isentropic density and, possibly, the water constituents
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 'in_mtg': self._in_mtg, 'in_s_old': self._in_s_old} 
		_outputs = {'out_s': self._out_s}
		if self._moist_on:
			_inputs.update({'in_qv': self._in_qv, 'in_qv_old': self._in_qv_old,
							'in_qc': self._in_qc, 'in_qc_old': self._in_qc_old,
							'in_qr': self._in_qr, 'in_qr_old': self._in_qr_old})
			_outputs.update({'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr})

		# Instantiate the stencil stepping the isentropic density and, possibly, the water constituents
		nb = self.nb
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		self._stencil_stepping_by_neglecting_vertical_advection_unstaggered = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_unstaggered_defs,
			inputs = _inputs,
			global_inputs = {'dt': self._dt},
			outputs = _outputs,
			domain = gt.domain.Rectangle((nb, nb, 0), (nb + ni - 1, nb + nj - 1, nk - 1)), 
			mode = self._backend)

		# Instantiate the stencil stepping the x-velocity
		ni, nj, nk = mi - 2 * nb + 1, mj - 2 * nb, nz
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_x = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_velocity_x_defs,
			inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 'in_mtg': self._in_mtg, 
					  'in_u_old': self._in_u_old},
			global_inputs = {'dt': self._dt},
			outputs = {'out_u': self._out_u},
			domain = gt.domain.Rectangle((nb, nb, 0), (nb + ni - 1, nb + nj - 1, nk - 1)), 
			mode = self._backend)

		# Instantiate the stencil stepping the y-velocity
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb + 1, nz
		self._stencil_stepping_by_neglecting_vertical_advection_velocity_y = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_velocity_y_defs,
			inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 'in_mtg': self._in_mtg, 
					  'in_v_old': self._in_v_old},
			global_inputs = {'dt': self._dt},
			outputs = {'out_v': self._out_v},
			domain = gt.domain.Rectangle((nb, nb, 0), (nb + ni - 1, nb + nj - 1, nk - 1)), 
			mode = self._backend)

	def _stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(self, mi, mj):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py stencil stepping the solution 
		by neglecting vertical advection.

		Parameters
		----------
		mi : int
			:math:`x`-extent of an input array representing an :math:`x`-unstaggered field.
		mj : int
			:math:`y`-extent of an input array representing a :math:`y`-unstaggered field.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Instantiate a GT4Py Global representing the timestep, and the Numpy arrays
		# which represent the solution at the current time step
		super()._stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(mi, mj)

		# Infer the size of the input arrays shared among different stencils
		li = mi if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mi, nx)
		lj = mj if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mj, ny)

		# Allocate the Numpy arrays which represent the solution at the previous time step,
		# and which may be shared among different stencils
		self._in_s_old  = np.zeros((li, lj, nz), dtype = datatype)
		self._in_qr_old = np.zeros((li, lj, nz), dtype = datatype)

		# Allocate the Numpy arrays which represent the solution at the previous time step,
		# and which are not shared among different stencils
		self._in_u_old = np.zeros((mi+1,   mj, nz), dtype = datatype)
		self._in_v_old = np.zeros((  mi, mj+1, nz), dtype = datatype)
		if self._moist_on:
			self._in_qv_old = np.zeros((mi, mj, nz), dtype = datatype)
			self._in_qc_old = np.zeros((mi, mj, nz), dtype = datatype)

	def _stencils_stepping_by_neglecting_vertical_advection_set_inputs(self, dt, state, state_old = None):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils stepping the solution 
		by neglecting vertical advection.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
		"""
		# Shortcuts
		mi, mj = self._mi, self._mj
		qv_label = 'mass_fraction_of_water_vapor_in_air' 
		qc_label = 'mass_fraction_of_cloud_liquid_water_in_air'
		qr_label = 'mass_fraction_of_precipitation_water_in_air'

		# Update the time step, and the Numpy arrays representing the current solution
		super()._stencils_stepping_by_neglecting_vertical_advection_set_inputs(dt, state)
		
		# Update the Numpy arrays representing the solution at the previous time step
		# This should be done only if the old state is explicitly given, or if this is the first time step
		if state_old is not None:		# Old state explicitly given
			self._in_s_old[:mi, :mj, :] = \
				self.boundary.from_physical_to_computational_domain(state_old['air_isentropic_density'].values[:,:,:,0])
			self._in_u_old[:mi+1, :mj, :] = \
				self.boundary.from_physical_to_computational_domain(state_old['x_velocity'].values[:,:,:,0])
			self._in_v_old[:mi, :mj+1, :] = \
				self.boundary.from_physical_to_computational_domain(state_old['y_velocity'].values[:,:,:,0])

			if self._moist_on:
				self._in_qv_old[:mi, :mj, :] = \
					self.boundary.from_physical_to_computational_domain(state_old[qv_label].values[:,:,:,0])
				self._in_qc_old[:mi, :mj, :] = \
					self.boundary.from_physical_to_computational_domain(state_old[qc_label].values[:,:,:,0])
				self._in_qr_old[:mi, :mj, :] = \
					self.boundary.from_physical_to_computational_domain(state_old[qr_label].values[:,:,:,0])
		elif self._is_first_timestep:	# This is the first timestep
			self._in_s_old[:mi, :mj, :] = \
				self.boundary.from_physical_to_computational_domain(state['air_isentropic_density'].values[:,:,:,0])
			self._in_u_old[:mi+1, :mj, :] = \
				self.boundary.from_physical_to_computational_domain(state['x_velocity'].values[:,:,:,0])
			self._in_v_old[:mi, :mj+1, :] = \
				self.boundary.from_physical_to_computational_domain(state['y_velocity'].values[:,:,:,0])

			if self._moist_on:
				self._in_qv_old[:mi, :mj, :] = \
					self.boundary.from_physical_to_computational_domain(state[qv_label].values[:,:,:,0])
				self._in_qc_old[:mi, :mj, :] = \
					self.boundary.from_physical_to_computational_domain(state[qc_label].values[:,:,:,0])
				self._in_qr_old[:mi, mj:, :] = \
					self.boundary.from_physical_to_computational_domain(state[qr_label].values[:,:,:,0])

	def _stencil_stepping_by_neglecting_vertical_advection_unstaggered_defs(self, dt, in_s, in_u, in_v, in_mtg, in_s_old, 
																			in_qv = None, in_qc = None, in_qr = None, 
																			in_qv_old = None, in_qc_old = None, in_qr_old = None):
		"""
		GT4Py stencil advancing the isentropic density and, possibly, the water constituents via a centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time level.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time level.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time level.
		in_s : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time level.
		in_s_old : obj
			:class:`gridtools.Equation` representing the isentropic density at the previous time level.
		in_qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of water vapor at the current time level.
		in_qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water at the current time level.
		in_qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of precipitation water at the current time level.
		in_qv_old : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of water vapor at the previous time level.
		in_qc_old : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water at the previous time level.
		in_qr_old : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction of precipitation water at the previous time level.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_qv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass fraction of water vapour.
		out_qc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass fraction of cloud liquid water.
		out_qr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass fraction of precipitation water.
		"""
		# Shortcuts
		dx, dy = self._grid.dx, self._grid.dy

		# Declare indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Declare temporary and output fields
		tmp_u = gt.Equation()
		tmp_v = gt.Equation()
		out_s = gt.Equation()
		if self._moist_on:
			out_qv = gt.Equation()
			out_qc = gt.Equation()
			out_qr = gt.Equation()

		# Evaluate the numerical fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, _, _, _, _ = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg)
		else:
			flux_s_x, flux_s_y, _, _, _, _,	flux_qv_x, flux_qv_y, flux_qc_x, flux_qc_y, flux_qr_x, flux_qr_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_qv, in_qc, in_qr)

		# Interpolate the velocity components at the mass points
		tmp_u[i, j, k] = in_u[i, j, k] + in_u[i+1, j, k]
		tmp_v[i, j, k] = in_v[i, j, k] + in_v[i, j+1, k]

		# Step the isentropic density
		out_s[i, j, k] = in_s_old[i, j, k] - 2. * dt * ((flux_s_x[i+1, j, k] - flux_s_x[i, j, k]) / dx +
						 					            (flux_s_y[i, j+1, k] - flux_s_y[i, j, k]) / dy)

		if self._moist_on:
			# Step the mass fraction of water vapor
			out_qv[i, j, k] = in_qv_old[i, j, k] - dt * (tmp_u[i, j, k] * (flux_qv_x[i+1, j, k] - flux_qv_x[i, j, k]) / dx +
						 						  	 	 tmp_v[i, j, k] * (flux_qv_y[i, j+1, k] - flux_qv_y[i, j, k]) / dy)

			# Step the mass fraction of cloud liquid water
			out_qc[i, j, k] = in_qc_old[i, j, k] - dt * (tmp_u[i, j, k] * (flux_qc_x[i+1, j, k] - flux_qc_x[i, j, k]) / dx +
						 						  	 	 tmp_v[i, j, k] * (flux_qc_y[i, j+1, k] - flux_qc_y[i, j, k]) / dy)

			# Step the mass fraction of precipitation water
			out_qr[i, j, k] = in_qr_old[i, j, k] - dt * (tmp_u[i, j, k] * (flux_qr_x[i+1, j, k] - flux_qr_x[i, j, k]) / dx +
						 						  	 	 tmp_v[i, j, k] * (flux_qr_y[i, j+1, k] - flux_qr_y[i, j, k]) / dy)

		if not self._moist_on:
			return out_s
		else:
			return out_s, out_qv, out_qc, out_qr

	def _stencil_stepping_by_neglecting_vertical_advection_velocity_x_defs(self, dt, in_s, in_u, in_v, in_mtg, in_u_old): 
		"""
		GT4Py stencil advancing the :math:`x`-velocity via a centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time level.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time level.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time level.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time level.
		in_u_old : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the previous time level.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the stepped :math:`x`-velocity.
		"""
		# Shortcuts
		dx, dy = self._grid.dx, self._grid.dy

		# Declare indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Declare the output field
		out_u = gt.Equation()

		# Evaluate the numerical fluxes
		_, _, flux_u_x, flux_u_y, _, _ = self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg)

		# Step the x-velocity
		out_u[i, j, k] = in_u_old[i, j, k] \
						 - 2. * dt * (in_u[i, j, k] * (flux_u_x[i, j, k] - flux_u_x[i-1, j, k]) / dx +
						 	     	  0.25 * (in_v[i-1, j, k] + in_v[i, j, k] + in_v[i-1, j+1, k] + in_v[i, j+1, k]) * 
								 	  (flux_u_y[i, j+1, k] - flux_u_y[i, j, k]) / dy +
								 	  (in_mtg[i, j, k] - in_mtg[i-1, j, k]) / dx)

		return out_u

	def _stencil_stepping_by_neglecting_vertical_advection_velocity_y_defs(self, dt, in_s, in_u, in_v, in_mtg, in_v_old): 
		"""
		GT4Py stencil advancing the :math:`y`-velocity via a centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time level.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time level.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time level.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time level.
		in_v_old : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the previous time level.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the stepped :math:`y`-velocity.
		"""
		# Shortcuts
		dx, dy = self._grid.dx, self._grid.dy

		# Declare indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Declare the output field
		out_v = gt.Equation()

		# Evaluate the numerical fluxes
		_, _, _, _, flux_v_x, flux_v_y = self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg)

		# Step the y-velocity
		out_v[i, j, k] = in_v_old[i, j, k] \
						 - 2. * dt * (0.25 * (in_u[i, j-1, k] + in_u[i+1, j-1, k] + in_u[i, j, k] + in_u[i+1, j, k]) *
								 	  (flux_v_x[i+1, j, k] - flux_v_x[i, j, k]) / dx +
						 	     	  in_v[i, j, k] * (flux_v_y[i, j, k] - flux_v_y[i, j-1, k]) / dy + 
								 	  (in_mtg[i, j, k] - in_mtg[i, j-1, k]) / dy)

		return out_v

	def _stencil_stepping_by_coupling_physics_with_dynamics_defs(dt, in_w, 
																 in_s_now, in_s_prv, 
																 in_u_now, in_u_prv, 
																 in_v_now, in_v_prv,
															  	 qv_now = None, qv_prv = None, 
															  	 qc_now = None, qc_prv = None,
															  	 qr_now = None, qr_prv = None):
		"""
		GT4Py stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		in_w : array_like
			:class:`numpy.ndarray` representing the vertical velocity, i.e., the change over time in potential temperature.
		in_s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		in_s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		in_u_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-velocity.
		in_u_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-velocity.
		in_v_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-velocity.
		in_v_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-velocity.
		in_qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current mass fraction of water vapor.
		in_qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of water vapor.
		in_qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current mass fraction of cloud liquid water.
		in_qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of cloud liquid water.
		in_qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current mass fraction of precipitation water.
		in_qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional mass fraction of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		out_u : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-velocity.
		out_v : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-velocity.
		out_qv : `obj`, optional 
			:class:`gridtools.Equation` representing the updated mass fraction of water vapor.
		out_qc : `obj`, optional 
			:class:`gridtools.Equation` representing the updated mass fraction of cloud liquid water.
		out_qr : `obj`, optional 
			:class:`gridtools.Equation` representing the updated mass fraction of precipitation water.
		"""
		### TODO ###

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
		self._in_s_prv  = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_h      = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_qr     = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_qr_prv = np.zeros((nx, ny, nz  ), dtype = datatype)
		self._in_vt     = np.zeros((nx, ny, nz  ), dtype = datatype)

		# Allocate the Numpy arrays which will be shared accross different stencils
		self._tmp_s_tnd  = np.zeros((nx, ny, nz), dtype = datatype)
		self._tmp_qr_tnd = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencils' outputs
		self._out_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the GT4Py stencil in charge of computing the slow tendencies
		self._stencil_computing_slow_tendencies = gt.NGStencil(
			definitions_func = self._stencil_computing_slow_tendencies_defs,
			inputs           = {'in_s_old': self._in_s, 'in_s_prv': self._in_s_prv,
								'in_qr_old': self._in_qr, 'in_qr_prv': self._in_qr_prv},
			global_inputs    = {'dt': self._dt},
			outputs          = {'out_s_tnd': self._tmp_s_tnd, 'out_qr_tnd': self._tmp_qr_tnd},
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

	def _stencil_computing_slow_tendencies_defs(self, dt, in_s_old, in_s_prv, in_qr_old, in_qr_prv):
		"""
		GT4Py stencil computing the slow tendencies required to resolve rain sedimentation.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the large timestep.
		in_s_old : obj
			:class:`gridtools.Equation` representing the old isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density.
		in_qr_old : obj
			:class:`gridtools.Equation` representing the old mass fraction of precipitation water.
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
		out_s_tnd[i, j, k]  = 0.5 * (in_s_prv[i, j, k] - in_s_old[i, j, k]) / dt
		out_qr_tnd[i, j, k] = 0.5 * (in_qr_prv[i, j, k] - in_qr_old[i, j, k]) / dt

		return out_s_tnd, out_qr_tnd

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
		tmp_qr[i, j, k] = tmp_qr_st[i, j, k] - dts * tmp_dfdz[i, j, k] / in_rho[i, j, k]
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
