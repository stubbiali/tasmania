"""
This module contains:
	Centered(IsentropicPrognostic)
	ForwardEuler(IsentropicPrognostic)
	RK2(IsentropicPrognostic)
	RK3COSMO(RK2)
	RK3(IsentropicPrognostic)
"""
import numpy as np

import gridtools as gt
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.utils.data_utils import make_state

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


# Convenient aliases
mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


class Centered(IsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.prognostic_isentropic.PrognosticIsentropic`
	to implement a centered time-integration scheme which takes over the
	prognostic part of the three-dimensional moist isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, diagnostics, horizontal_boundary_conditions,
						 horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
						 sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
						 raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be re-directed when the corresponding forward methods
		# are invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_motion = None
		self._stencil_computing_slow_tendencies = None
		self._stencil_ensuring_vertical_cfl_is_obeyed = None
		self._stencil_stepping_by_integrating_sedimentation_flux = None

		# Boolean flag to quickly assess whether we are within the first time step
		self._is_first_timestep = True

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 1

	def step_neglecting_vertical_motion(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_stepping_by_neglecting_vertical_motion is None:
			self._stencil_stepping_by_neglecting_vertical_motion_initialize(
				raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# Run the stencil's compute function
		self._stencil_stepping_by_neglecting_vertical_motion.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))

		# Bring the updated momenta back to the original dimensions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))

		# Bring the updated water constituents back to the original dimensions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))

		# Apply the boundary conditions
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])
		if self._moist_on:
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'time': raw_state['time'] + dt,
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		# Keep track of the current state for the next time step
		self._in_s_old[:mi, :mj, :]  = self._in_s[:mi, :mj, :]
		self._in_su_old[:mi, :mj, :] = self._in_su[:mi, :mj, :]
		self._in_sv_old[:mi, :mj, :] = self._in_sv[:mi, :mj, :]
		if self._moist_on:
			self._in_sqv_old[:mi, :mj, :] = self._in_sqv[:mi, :mj, :]
			self._in_sqc_old[:mi, :mj, :] = self._in_sqc[:mi, :mj, :]
			self._in_sqr_old[:mi, :mj, :] = self._in_sqr[:mi, :mj, :]

		# At this point, the first time step is surely over
		self._is_first_timestep = False

		return raw_state_new

	def step_integrating_vertical_advection(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time by integrating the vertical advection, i.e.,
		by accounting for the change over time in potential temperature.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.
		"""
		raise NotImplementedError()

	def step_integrating_sedimentation_flux(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the mass fraction of precipitation water by taking
		the sedimentation into account. For the sake of numerical stability,
		a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a time step which may be smaller than that specified by the user.
		"""
		# Shortcuts
		nb = self._sflux.nb
		pt = raw_state_now['air_pressure_on_interface_levels'][0, 0, 0]

		# The first time this method is invoked, initialize the underlying GT4Py stencils
		if self._stencil_stepping_by_integrating_sedimentation_flux is None:
			self._stencils_stepping_by_integrating_sedimentation_flux_initialize()

		# Compute the smaller timestep
		dts = dt / float(self._sedimentation_substeps)

		# Update the attributes which serve as inputs to the GT4Py stencils
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._dt.value	= dt.total_seconds()
		self._dts.value = dts.total_seconds()
		self._in_s[:nx, :ny, :]		= self._in_s_old[:nx, :ny, :]
		self._in_s_prv[:nx, :ny, :] = raw_state_prv['air_isentropic_density'][...]
		self._water_constituent_diagnostic.get_mass_fraction_of_water_constituent_in_air(
			self._in_s_old[:nx, :ny, :], self._in_sqr_old[:nx, :ny, :], self._in_qr)
		self._in_qr_prv[...] = raw_state_prv[mf_pw][...]

		# Compute the slow tendencies from the large timestepping
		self._stencil_computing_slow_tendencies.compute()

		# Advance the solution
		self._in_s[:nx, :ny, nb:] = raw_state_now['air_isentropic_density'][:, :, nb:]
		self._in_qr[:, :, nb:]	  = raw_state_now[mf_pw][:, :, nb:]

		# Set the upper layers of the output fields
		self._out_s[:nx, :ny, :nb] = raw_state_prv['air_isentropic_density'][:, :, :nb]
		self._out_qr[:, :, :nb]    = self._in_qr[:, :, :nb]

		# Initialize the new raw state
		raw_state_new = {
			'time': raw_state_now['time'] + dt,
			'air_density': self._in_rho,
			'height_on_interface_levels': self._in_h,
			mf_pw: self._in_qr,
		}
		state_new_units = {
			'air_density': 'kg m^-3',
			'height_on_interface_levels': 'm',
			mf_pw: 'g g^-1',
		}

		# Convert new raw state in state dictionary
		state_new = make_state(raw_state_new, self._grid, state_new_units)

		# Initialize the arrays storing the precipitation and the accumulated precipitation
		precipitation = np.zeros((nx, ny, 1), dtype=self._dtype)
		accumulated_precipitation = np.zeros((nx, ny, 1), dtype=self._dtype)
		if raw_state_now.get('accumulated_precipitation', None) is not None:
			accumulated_precipitation[...] = \
				raw_state_now['accumulated_precipitation'][...]

		# Perform the time-splitting procedure
		for n in range(self._sedimentation_substeps):
			# Diagnose the geometric height of the interface vertical levels
			self._in_h[...] = self._diagnostics.get_height(self._in_s[:nx, :ny, :], pt)

			# Diagnose the air density
			self._in_rho[...] = self._diagnostics.get_air_density(self._in_s[:nx, :ny, :],
																	  self._in_h)

			# Compute the raindrop fall velocity
			vt_dict = self._fall_velocity_diagnostic(state_new)
			self._in_vt[...] = \
				vt_dict['raindrop_fall_velocity'].to_units('m s^-1').values[...]

			# Make sure the vertical CFL is obeyed
			self._stencil_ensuring_vertical_cfl_is_obeyed.compute()
			self._in_vt[...] = self._out_vt[...]

			# Compute the precipitation and the accumulated precipitation
			# Note: the precipitation is accumulated only on the time interval
			# from the current to the next time level
			rho_water = self._physical_constants['density_of_liquid_water']
			ppt = self._in_rho[:, :, -1:] * self._in_qr[:, :, -1:] * self._in_vt[:, :, -1:] * \
				  self._dts.value / rho_water
			precipitation[...] = ppt[...] / self._dts.value * 3.6e6
			if n >= self._sedimentation_substeps / 2:
				accumulated_precipitation[...] += ppt[...] * 1.e3

			# Perform a small timestep
			self._stencil_stepping_by_integrating_sedimentation_flux.compute()

			# Advance the solution
			self._in_s[:, :, nb:]  = self._out_s[:, :, nb:]
			self._in_qr[:, :, nb:] = self._out_qr[:, :, nb:]

		# Instantiate raw output state
		raw_state_out = {
			mf_pw: self._in_qr,
			'precipitation': precipitation,
			'accumulated_precipitation': accumulated_precipitation,
		}

		return raw_state_out

	def _stencil_stepping_by_neglecting_vertical_motion_initialize(
		self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing a time-integration centered scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_outputs()

		# Set the stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_mtg': self._in_mtg, 'in_su': self._in_su, 'in_sv': self._in_sv,
				   'in_s_old': self._in_s_old, 'in_su_old': self._in_su_old,
				   'in_sv_old': self._in_sv_old}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_old': self._in_sqv_old,
							'in_sqc': self._in_sqc, 'in_sqc_old': self._in_sqc_old,
							'in_sqr': self._in_sqr, 'in_sqr_old': self._in_sqr_old})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Set the stencil's computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb+ni-1, nb+nj-1, nk-1))
		_mode = self._backend

		# Instantiate the stencil
		self._stencil_stepping_by_neglecting_vertical_motion = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py
		stencil stepping the solution by neglecting vertical advection.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Instantiate a GT4Py Global representing the time step, and the Numpy arrays
		# which represent the solution at the current time step
		super()._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Determine the size of the arrays which will serve as stencils'
		# inputs and outputs. These arrays may be shared with the stencil
		# in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the Numpy arrays which represent the solution at the previous time step,
		# and which may be shared among different stencils
		self._in_s_old	= np.zeros((li, lj, nz), dtype=dtype)
		self._in_sqr_old = np.zeros((li, lj, nz), dtype=dtype)

		# Allocate the Numpy arrays which represent the solution at the previous time step,
		# and which are not shared among different stencils
		self._in_su_old = np.zeros((mi, mj, nz), dtype=dtype)
		self._in_sv_old = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_old = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc_old = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_set_inputs(
		self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding the vertical advection.
		"""
		# Update the time step, and the Numpy arrays representing
		# the current solution
		super()._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# At the first iteration, update the Numpy arrays representing
		# the solution at the previous time step
		if self._is_first_timestep:
			self._in_s_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['air_isentropic_density'])
			self._in_su_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['x_momentum_isentropic'])
			self._in_sv_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['y_momentum_isentropic'])

			if self._moist_on:
				self._in_sqv_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_water_vapor'])
				self._in_sqc_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_cloud_liquid_water'])
				self._in_sqr_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_precipitation_water'])

	def _stencil_stepping_by_neglecting_vertical_motion_defs(
		self, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
		in_s_old, in_su_old, in_sv_old,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_sqv_old=None, in_sqc_old=None, in_sqr_old=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil implementing the centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the current time.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity
			at the current time.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity
			at the current time.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential
			at the current time.
		in_su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the current time.
		in_sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the current time.
		in_s_old : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the previous time level.
		in_su_old : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the previous time level.
		in_sv_old : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the previous time level.
		in_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			water vapor at the current time.
		in_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			cloud water at the current time.
		in_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			precipitation water at the current time.
		in_sqv_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			water vapor at the previous time level.
		in_sqc_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			cloud water at the previous time level.
		in_sqr_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			precipitation water at the previous time level.
		in_s_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			isentropic density.
		in_su_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the :math:`x`-momentum.
		in_sv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the :math:`y`-momentum.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_su : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_sv : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		out_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of water vapor.
		out_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of cloud water.
		out_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)
		k = gt.Index(axis=2)

		# Instantiate the output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the horizontal fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
							s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
							in_sqv, in_sqc, in_sqr, in_s_tnd, in_su_tnd, in_sv_tnd,
							in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s_old[i, j] \
						  - 2. * dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
									   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			out_s[i, j] = in_s_old[i, j] \
						  - 2. * dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
									   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
									   in_s_tnd[i, j])

		# Advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su_old[i, j] \
						   - 2. * dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
										(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy) \
						   - dt * in_s[i, j] * (in_mtg[i+1, j] - in_mtg[i-1, j]) / dx
		else:
			out_su[i, j] = in_su_old[i, j] \
						   - 2. * dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
										(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
										in_su_tnd[i, j]) \
						   - dt * in_s[i, j] * (in_mtg[i+1, j] - in_mtg[i-1, j]) / dx

		# Advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv_old[i, j] \
						   - 2. * dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
										(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy) \
						   - dt * in_s[i, j] * (in_mtg[i, j+1] - in_mtg[i, j-1]) / dy
		else:
			out_sv[i, j] = in_sv_old[i, j] \
						   - 2. * dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
										(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
										in_sv_tnd[i, j]) \
						   - dt * in_s[i, j] * (in_mtg[i, j+1] - in_mtg[i, j-1]) / dy

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv_old[i, j] \
								- 2. * dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
											 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv_old[i, j] \
								- 2. * dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
											 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
											 in_s[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc_old[i, j] \
								- 2. * dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
											 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc_old[i, j] \
								- 2. * dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
											 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
											 in_s[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr_old[i, j] \
								- 2. * dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
											 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr_old[i, j] \
								- 2. * dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
											 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
											 in_s[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr

	def _stencils_stepping_by_integrating_sedimentation_flux_initialize(self):
		"""
		Initialize the GT4Py stencils in charge of stepping the mass fraction
		of precipitation water by integrating the sedimentation flux.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._sflux.nb
		dtype = self._dtype

		# Allocate the GT4Py Globals which represent the small and large time step
		self._dts = gt.Global()
		if not hasattr(self, '_dt'):
			self._dt = gt.Global()

		# Allocate the Numpy arrays which will serve as stencils' inputs
		self._in_rho	= np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_s_prv	= np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_h		= np.zeros((nx, ny, nz+1), dtype=dtype)
		self._in_qr		= np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_qr_prv = np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_vt		= np.zeros((nx, ny, nz	), dtype=dtype)

		# Allocate the Numpy arrays which will be shared across different stencils
		self._tmp_s_tnd  = np.zeros((nx, ny, nz), dtype=dtype)
		self._tmp_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the Numpy arrays which will serve as stencils' outputs
		self._out_vt = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Initialize the GT4Py stencil in charge of computing the slow tendencies
		self._stencil_computing_slow_tendencies = gt.NGStencil(
			definitions_func = self._stencil_computing_slow_tendencies_defs,
			inputs			 = {'in_s_old': self._in_s[:nx, :ny, :], 'in_s_prv': self._in_s_prv,
								'in_qr_old': self._in_qr, 'in_qr_prv': self._in_qr_prv},
			global_inputs	 = {'dt': self._dt},
			outputs			 = {'out_s_tnd': self._tmp_s_tnd, 'out_qr_tnd': self._tmp_qr_tnd},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

		# Initialize the GT4Py stencil ensuring that the vertical CFL condition is fulfilled
		self._stencil_ensuring_vertical_cfl_is_obeyed = gt.NGStencil(
			definitions_func = self._stencil_ensuring_vertical_cfl_is_obeyed_defs,
			inputs			 = {'in_h': self._in_h, 'in_vt': self._in_vt},
			global_inputs	 = {'dts': self._dts},
			outputs			 = {'out_vt': self._out_vt},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

		# Initialize the GT4Py stencil in charge of actually stepping the solution
		# by integrating the sedimentation flux
		self._stencil_stepping_by_integrating_sedimentation_flux = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_integrating_sedimentation_flux_defs,
			inputs			 = {'in_rho': self._in_rho, 'in_s': self._in_s[:nx, :ny, :],
								'in_h': self._in_h, 'in_qr': self._in_qr, 'in_vt': self._in_vt,
								'in_s_tnd': self._tmp_s_tnd, 'in_qr_tnd': self._tmp_qr_tnd},
			global_inputs	 = {'dts': self._dts},
			outputs			 = {'out_s': self._out_s[:nx, :ny, :], 'out_qr': self._out_qr},
			domain			 = gt.domain.Rectangle((0, 0, nb), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

	@staticmethod
	def _stencil_computing_slow_tendencies_defs(dt, in_s_old, in_s_prv,
												in_qr_old, in_qr_prv):
		"""
		GT4Py stencil computing the slow tendencies of isentropic density
		and mass fraction of precipitation rain.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the large timestep.
		in_s_old : obj
			:class:`gridtools.Equation` representing the old
			isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional
			isentropic density.
		in_qr_old : obj
			:class:`gridtools.Equation` representing the old
			mass fraction of precipitation water.
		in_qr_prv : obj
			:class:`gridtools.Equation` representing the provisional
			mass fraction of precipitation water.

		Return
		------
		out_s_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency
			of the isentropic density.
		out_qr_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency
			of the mass fraction of precipitation water.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s_tnd  = gt.Equation()
		out_qr_tnd = gt.Equation()

		# Computations
		out_s_tnd[i, j, k]	= 0.5 * (in_s_prv[i, j, k] - in_s_old[i, j, k]) / dt
		out_qr_tnd[i, j, k] = 0.5 * (in_qr_prv[i, j, k] - in_qr_old[i, j, k]) / dt

		return out_s_tnd, out_qr_tnd

	@staticmethod
	def _stencil_ensuring_vertical_cfl_is_obeyed_defs(dts, in_h, in_vt):
		"""
		GT4Py stencil ensuring that the vertical CFL condition is fulfilled.
		This is achieved by clipping the raindrop fall velocity field:
		if a cell does not satisfy the CFL constraint, the vertical velocity
		at that cell is reduced so that the local CFL number equals 0.95.

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
		# Indices
		k = gt.Index(axis=2)

		# Temporary and output fields
		tmp_cfl = gt.Equation()
		out_vt	= gt.Equation()

		# Computations
		tmp_cfl[k] = in_vt[k] * dts / (in_h[k] - in_h[k+1])
		out_vt[k]  = (tmp_cfl[k] < 0.95) * in_vt[k] + \
					 (tmp_cfl[k] > 0.95) * 0.95 * (in_h[k] - in_h[k+1]) / dts

		return out_vt

	def _stencil_stepping_by_integrating_sedimentation_flux_defs(
		self, dts, in_rho, in_s, in_h, in_qr, in_vt, in_s_tnd, in_qr_tnd):
		"""
		GT4Py stencil stepping the isentropic density and the mass fraction
		of precipitation water by integrating the precipitation flux.

		Parameters
		----------
		dts : obj
			:class:`gridtools.Global` representing the small timestep.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_s : obj
			:class:`gridtools.Equation` representing the air isentropic density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height
			of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the input mass fraction
			of precipitation water.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		in_s_tnd : obj
			:class:`gridtools.Equation` representing the slow tendency of
			isentropic density.
		in_qr_tnd : obj
			:class:`gridtools.Equation` representing the slow tendency of
			mass fraction of precipitation water.

		Return
		------
		out_s : obj
			:class:`gridtools.Equation` representing the output isentropic density.
		out_qr : obj
			:class:`gridtools.Equation` representing the output mass fraction of
			precipitation water.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output fields
		tmp_qr_st = gt.Equation()
		tmp_qr	  = gt.Equation()
		out_s	  = gt.Equation()
		out_qr	  = gt.Equation()

		# Update isentropic density
		out_s[i, j, k] = in_s[i, j, k] + dts * in_s_tnd[i, j, k]

		# Update mass fraction of precipitation water
		tmp_dfdz = self._sflux(k, in_rho, in_h, in_qr, in_vt)
		tmp_qr_st[i, j, k] = in_qr[i, j, k] + dts * in_qr_tnd[i, j, k]
		tmp_qr[i, j, k] = tmp_qr_st[i, j, k] + dts * tmp_dfdz[i, j, k] / in_rho[i, j, k]
		out_qr[i, j, k] = (tmp_qr[i, j, k] > 0.) * tmp_qr[i, j, k] + \
						  (tmp_qr[i, j, k] < 0.) * \
						  ((tmp_qr_st[i, j, k] > 0.) * tmp_qr_st[i, j, k])

		return out_s, out_qr


class ForwardEuler(IsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.prognostic_isentropic.PrognosticIsentropic`
	to implement the forward Euler time-integration scheme which takes over the
	prognostic part of the three-dimensional moist isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, diagnostics, horizontal_boundary_conditions,
						 horizontal_flux_scheme, adiabatic_flow,	vertical_flux_scheme,
						 sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
						 raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be re-directed when the corresponding forward methods
		# are invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_motion_first = None
		self._stencil_stepping_by_neglecting_vertical_motion_second = None
		self._stencil_computing_slow_tendencies = None
		self._stencil_ensuring_vertical_cfl_is_obeyed = None
		self._stencil_stepping_by_integrating_sedimentation_flux = None

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 1

	def step_neglecting_vertical_motion(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_stepping_by_neglecting_vertical_motion_first is None:
			self._stencils_stepping_by_neglecting_vertical_motion_initialize(
				raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# Run the compute function of the stencil stepping the isentropic density
		# and the water constituents, and providing provisional values for the momenta
		self._stencil_stepping_by_neglecting_vertical_motion_first.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated water constituents back to the original dimensions,
		# and enforce the boundary conditions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Compute the provisional isentropic density; this may be scheme-dependent
		if self._hflux_scheme in ['upwind', 'centered', 'fifth_order_upwind']:
			s_prov = s_new
		elif self._hflux_scheme in ['maccormack']:
			s_prov = .5 * (raw_state['air_isentropic_density'] + s_new)
		else:
			raise ValueError('Unknown flux scheme.')

		# Diagnose the Montgomery potential from the provisional isentropic density
		pt = raw_state['air_pressure_on_interface_levels'][0, 0, 0]
		_, _, mtg_prv, _ = self._diagnostics.get_diagnostic_variables(s_prov, pt)

		# Extend the updated isentropic density and Montgomery potential to
		# accommodate the horizontal boundary conditions
		self._in_s_prv[:mi, :mj, :]   = \
			self._hboundary.from_physical_to_computational_domain(s_prov)
		self._in_mtg_prv[:mi, :mj, :] = \
			self._hboundary.from_physical_to_computational_domain(mtg_prv)

		# Run the compute function of the stencil stepping the momenta
		self._stencil_stepping_by_neglecting_vertical_motion_second.compute()

		# Bring the updated momenta back to the original dimensions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))

		# Enforce the boundary conditions on the momenta
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		# Instantiate the output state
		raw_state_new = {
			'time': raw_state['time'] + dt,
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def step_integrating_vertical_advection(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time by integrating the vertical advection, i.e.,
		by accounting for the change over time in potential temperature.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.
		"""
		raise NotImplementedError()

	def step_integrating_sedimentation_flux(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the mass fraction of precipitation water by taking
		the sedimentation into account. For the sake of numerical stability,
		a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a time step which may be smaller than that specified by the user.
		"""
		# Shortcuts
		nb = self._sflux.nb
		pt = raw_state_now['air_pressure_on_interface_levels'][0, 0, 0]

		# The first time this method is invoked, initialize the underlying GT4Py stencils
		if self._stencil_stepping_by_integrating_sedimentation_flux is None:
			self._stencils_stepping_by_integrating_sedimentation_flux_initialize()

		# Compute the smaller timestep
		dts = dt / float(self._sedimentation_substeps)

		# Update the attributes which serve as inputs to the GT4Py stencils
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._dt.value	= dt.total_seconds()
		self._dts.value = dts.total_seconds()
		self._in_s[:nx, :ny, :]		= raw_state_now['air_isentropic_density'][...]
		self._in_s_prv[:nx, :ny, :] = raw_state_prv['air_isentropic_density'][...]
		self._in_qr[...]	 = raw_state_now[mf_pw][...]
		self._in_qr_prv[...] = raw_state_prv[mf_pw][...]

		# Compute the slow tendencies from the large timestepping
		self._stencil_computing_slow_tendencies.compute()

		# Set the upper layers of the output fields
		self._out_s[:nx, :ny, :nb] = raw_state_prv['air_isentropic_density'][:, :, :nb]
		self._out_qr[:, :, :nb]    = raw_state_prv[mf_pw][:, :, :nb]

		# Initialize new raw state
		raw_state_new = {
			'time': raw_state_now['time'] + dt,
			'air_density': self._in_rho,
			'height_on_interface_levels': self._in_h,
			mf_pw: self._in_qr,
		}
		state_new_units = {
			'air_density': 'kg m^-3',
			'height_on_interface_levels': 'm',
			mf_pw: 'g g^-1',
		}

		# Convert new raw state in state dictionary
		state_new = make_state(raw_state_new, self._grid, state_new_units)

		# Initialize the arrays storing the precipitation and the accumulated precipitation
		precipitation = np.zeros((nx, ny, 1), dtype=self._dtype)
		accumulated_precipitation = np.zeros((nx, ny, 1), dtype=self._dtype)
		if raw_state_now.get('accumulated_precipitation', None) is not None:
			accumulated_precipitation[...] = raw_state_now['accumulated_precipitation'][...]

		# Perform the time-splitting procedure
		for n in range(self._sedimentation_substeps):
			# Diagnose the geometric height of the interface vertical levels
			self._in_h[...] = self._diagnostics.get_height(self._in_s[:nx, :ny, :], pt)

			# Diagnose the air density
			self._in_rho[...] = self._diagnostics.get_air_density(self._in_s[:nx, :ny, :],
																  self._in_h)

			# Compute the raindrop fall velocity
			vt_dict = self._fall_velocity_diagnostic(state_new)
			self._in_vt[...] = \
				vt_dict['raindrop_fall_velocity'].to_units('m s^-1').values[...]

			# Make sure the vertical CFL is obeyed
			self._stencil_ensuring_vertical_cfl_is_obeyed.compute()
			self._in_vt[...] = self._out_vt[...]

			# Compute the precipitation and the accumulated precipitation
			# Note: the precipitation is accumulated only on the time interval
			# from the current to the next time level
			rho_water = self._physical_constants['density_of_liquid_water']
			ppt = self._in_rho[:, :, -1:] * self._in_qr[:, :, -1:] * self._in_vt[:, :, -1:] * \
				  self._dts.value / rho_water
			precipitation[...] = ppt[...] / self._dts.value * 3.6e6
			if n >= self._sedimentation_substeps / 2:
				accumulated_precipitation[...] += ppt[...] * 1.e3

			# Perform a small timestep
			self._stencil_stepping_by_integrating_sedimentation_flux.compute()

			# Advance the solution
			self._in_s[:, :, nb:]  = self._out_s[:, :, nb:]
			self._in_qr[:, :, nb:] = self._out_qr[:, :, nb:]

		# Instantiate raw output state
		raw_state_out = {
			mf_pw: self._in_qr,
			'precipitation': precipitation,
			'accumulated_precipitation': accumulated_precipitation,
		}

		return raw_state_out

	def _stencils_stepping_by_neglecting_vertical_motion_initialize(
		self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing a time-integration centered scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store provisional (i.e., temporary) fields
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_temporaries()

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_outputs()

		# Set the stencils' computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the first stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_mtg': self._in_mtg, 'in_su': self._in_su, 'in_sv': self._in_sv}
		_outputs = {'out_s': self._out_s, 'out_su': self._tmp_su, 'out_sv': self._tmp_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqc': self._in_sqc,
							'in_sqr': self._in_sqr})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Instantiate the first stencil
		self._stencil_stepping_by_neglecting_vertical_motion_first = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_first_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

		# Instantiate the second stencil
		self._stencil_stepping_by_neglecting_vertical_motion_second = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_second_defs,
			inputs			 = {'in_s': self._in_s_prv, 'in_mtg': self._in_mtg_prv,
								'in_su': self._tmp_su, 'in_sv': self._tmp_sv},
			global_inputs	 = {'dt': self._dt},
			outputs			 = {'out_su': self._out_su, 'out_sv': self._out_sv},
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_temporaries(self):
		"""
		Allocate the Numpy arrays which will store temporary fields to be shared
		between the stencils stepping the solution by neglecting vertical advection.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Determine the size of the arrays
		# Even if these arrays will not be shared with the stencil in charge
		# of integrating the vertical advection, they should be treated as they were
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the arrays
		self._tmp_su	 = np.zeros((li, lj, nz), dtype=self._dtype)
		self._tmp_sv	 = np.zeros((li, lj, nz), dtype=self._dtype)
		self._in_s_prv	 = np.zeros((li, lj, nz), dtype=self._dtype)
		self._in_mtg_prv = np.zeros((li, lj, nz), dtype=self._dtype)

	def _stencil_stepping_by_neglecting_vertical_motion_first_defs(
		self, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the isentropic density and the water constituents
		via the forward Euler scheme. Further, it computes provisional values for
		the momenta, i.e., it updates the momenta disregarding the forcing terms
		involving the Montgomery potential.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the current time level.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity
			at the current time level.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity
			at the current time level.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential
			at the current time level.
		in_su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the current time level.
		in_sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the current time level.
		in_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of water vapor at the current time level.
		in_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			mass of cloud water at the current time level.
		in_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			mass of precipitation water at the current time level.
		in_s_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the isentropic density.
		in_su_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the :math:`x`-momentum.
		in_sv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the :math:`y`-momentum.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped
			isentropic density.
		out_su : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`x`-momentum.
		out_sv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`y`-momentum.
		out_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of water vapour.
		out_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of cloud water.
		out_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of precipitation water.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
							s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
							in_sqv, in_sqc, in_sqr,	in_s_tnd, in_su_tnd, in_sv_tnd,
							in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
											 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
											 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
											 in_s_tnd[i, j])

		# Advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
											   in_su_tnd[i, j])

		# Advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
											   in_sv_tnd[i, j])

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
									  in_s[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
									  in_s[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
									  in_s[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr

	def _stencil_stepping_by_neglecting_vertical_motion_second_defs(
		self, dt, in_s, in_mtg, in_su, in_sv):
		"""
		GT4Py stencil stepping the momenta via a one-time-level scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the stepped
			isentropic density.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery
			potential diagnosed from the stepped isentropic density.
		in_su : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`x`-momentum.
		in_sv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`y`-momentum.

		Returns
		-------
		out_su : obj
			:class:`gridtools.Equation` representing the stepped
			:math:`x`-momentum.
		out_sv : obj
			:class:`gridtools.Equation` representing the stepped
			:math:`y`-momentum.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output fields
		out_su = gt.Equation()
		out_sv = gt.Equation()

		# Computations
		out_su[i, j] = in_su[i, j] - \
					   dt * 0.5 * in_s[i, j] * (in_mtg[i+1, j] - in_mtg[i-1, j]) / dx
		out_sv[i, j] = in_sv[i, j] - \
					   dt * 0.5 * in_s[i, j] * (in_mtg[i, j+1] - in_mtg[i, j-1]) / dy

		return out_su, out_sv

	def _stencils_stepping_by_integrating_sedimentation_flux_initialize(self):
		"""
		Initialize the GT4Py stencils in charge of stepping the mass fraction
		of precipitation water by integrating the sedimentation flux.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._sflux.nb
		dtype = self._dtype

		# Allocate the GT4Py Globals which represent the small and large time step
		self._dts = gt.Global()
		if not hasattr(self, '_dt'):
			self._dt = gt.Global()

		# Allocate the Numpy arrays which will serve as stencils' inputs
		self._in_rho	= np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_h		= np.zeros((nx, ny, nz+1), dtype=dtype)
		self._in_qr		= np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_qr_prv = np.zeros((nx, ny, nz	), dtype=dtype)
		self._in_vt		= np.zeros((nx, ny, nz	), dtype=dtype)
		if not hasattr(self, '_in_s_prv'):
			self._in_s_prv = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the Numpy arrays which will be shared across different stencils
		self._tmp_s_tnd  = np.zeros((nx, ny, nz), dtype=dtype)
		self._tmp_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the Numpy arrays which will serve as stencils' outputs
		self._out_vt = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Initialize the GT4Py stencil in charge of computing the slow tendencies
		self._stencil_computing_slow_tendencies = gt.NGStencil(
			definitions_func = self._stencil_computing_slow_tendencies_defs,
			inputs			 = {'in_s': self._in_s[:nx, :ny, :], 'in_s_prv': self._in_s_prv,
								'in_qr': self._in_qr, 'in_qr_prv': self._in_qr_prv},
			global_inputs	 = {'dt': self._dt},
			outputs			 = {'out_s_tnd': self._tmp_s_tnd, 'out_qr_tnd': self._tmp_qr_tnd},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

		# Initialize the GT4Py stencil ensuring that the vertical CFL condition is fulfilled
		self._stencil_ensuring_vertical_cfl_is_obeyed = gt.NGStencil(
			definitions_func = self._stencil_ensuring_vertical_cfl_is_obeyed_defs,
			inputs			 = {'in_h': self._in_h, 'in_vt': self._in_vt},
			global_inputs	 = {'dts': self._dts},
			outputs			 = {'out_vt': self._out_vt},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

		# Initialize the GT4Py stencil in charge of actually stepping the solution
		# by integrating the sedimentation flux
		self._stencil_stepping_by_integrating_sedimentation_flux = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_integrating_sedimentation_flux_defs,
			inputs			 = {'in_rho': self._in_rho, 'in_s': self._in_s[:nx, :ny, :],
								'in_h': self._in_h, 'in_qr': self._in_qr, 'in_vt': self._in_vt,
								'in_s_tnd': self._tmp_s_tnd, 'in_qr_tnd': self._tmp_qr_tnd},
			global_inputs	 = {'dts': self._dts},
			outputs			 = {'out_s': self._out_s[:nx, :ny, :], 'out_qr': self._out_qr},
			domain			 = gt.domain.Rectangle((0, 0, nb), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

	@staticmethod
	def _stencil_computing_slow_tendencies_defs(dt, in_s, in_s_prv,
												in_qr, in_qr_prv):
		"""
		GT4Py stencil computing the slow tendencies of isentropic density
		and mass fraction of precipitation rain.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the large timestep.
		in_s : obj
			:class:`gridtools.Equation` representing the current
			isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional
			isentropic density.
		in_qr : obj
			:class:`gridtools.Equation` representing the current
			mass fraction of precipitation water.
		in_qr_prv : obj
			:class:`gridtools.Equation` representing the provisional
			mass fraction of precipitation water.

		Return
		------
		out_s_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency
			of the isentropic density.
		out_qr_tnd : obj :
			:class:`gridtools.Equation` representing the slow tendency
			of the mass fraction of precipitation water.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s_tnd  = gt.Equation()
		out_qr_tnd = gt.Equation()

		# Computations
		out_s_tnd[i, j, k]	= (in_s_prv[i, j, k] - in_s[i, j, k]) / dt
		out_qr_tnd[i, j, k] = (in_qr_prv[i, j, k] - in_qr[i, j, k]) / dt

		return out_s_tnd, out_qr_tnd

	@staticmethod
	def _stencil_ensuring_vertical_cfl_is_obeyed_defs(dts, in_h, in_vt):
		"""
		GT4Py stencil ensuring that the vertical CFL condition is fulfilled.
		This is achieved by clipping the raindrop fall velocity field:
		if a cell does not satisfy the CFL constraint, the vertical velocity
		at that cell is reduced so that the local CFL number equals 0.95.

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
		# Indices
		k = gt.Index(axis=2)

		# Temporary and output fields
		tmp_cfl = gt.Equation()
		out_vt	= gt.Equation()

		# Computations
		tmp_cfl[k] = in_vt[k] * dts / (in_h[k] - in_h[k+1])
		out_vt[k]  = (tmp_cfl[k] < 0.95) * in_vt[k] + \
					 (tmp_cfl[k] > 0.95) * 0.95 * (in_h[k] - in_h[k+1]) / dts

		return out_vt

	def _stencil_stepping_by_integrating_sedimentation_flux_defs(
		self, dts, in_rho, in_s, in_h, in_qr, in_vt, in_s_tnd, in_qr_tnd):
		"""
		GT4Py stencil stepping the isentropic density and the mass fraction
		of precipitation water by integrating the precipitation flux.

		Parameters
		----------
		dts : obj
			:class:`gridtools.Global` representing the small timestep.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_s : obj
			:class:`gridtools.Equation` representing the air isentropic density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height
			of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the input mass fraction
			of precipitation water.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		in_s_tnd : obj
			:class:`gridtools.Equation` representing the slow tendency of
			isentropic density.
		in_qr_tnd : obj
			:class:`gridtools.Equation` representing the slow tendency of
			mass fraction of precipitation water.

		Return
		------
		out_s : obj
			:class:`gridtools.Equation` representing the output isentropic density.
		out_qr : obj
			:class:`gridtools.Equation` representing the output mass fraction of
			precipitation water.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output fields
		tmp_qr_st = gt.Equation()
		tmp_qr	  = gt.Equation()
		out_s	  = gt.Equation()
		out_qr	  = gt.Equation()

		# Update isentropic density
		out_s[i, j, k] = in_s[i, j, k] + dts * in_s_tnd[i, j, k]

		# Update mass fraction of precipitation water
		tmp_dfdz = self._sflux(k, in_rho, in_h, in_qr, in_vt)
		tmp_qr_st[i, j, k] = in_qr[i, j, k] + dts * in_qr_tnd[i, j, k]
		tmp_qr[i, j, k] = tmp_qr_st[i, j, k] + dts * tmp_dfdz[i, j, k] / in_rho[i, j, k]
		out_qr[i, j, k] = (tmp_qr[i, j, k] > 0.) * tmp_qr[i, j, k] + \
						  (tmp_qr[i, j, k] < 0.) * \
						  ((tmp_qr_st[i, j, k] > 0.) * tmp_qr_st[i, j, k])

		return out_s, out_qr


class RK2(IsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.prognostic_isentropic.PrognosticIsentropic`
	to implement the two-stages, second-order Runge-Kutta scheme which takes over the
	prognostic part of the three-dimensional moist isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, diagnostics, horizontal_boundary_conditions,
						 horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
						 sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
						 raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be re-directed when the corresponding forward methods
		# are invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_motion = None
		self._stencil_computing_slow_tendencies = None
		self._stencil_ensuring_vertical_cfl_is_obeyed = None
		self._stencil_stepping_by_integrating_sedimentation_flux = None

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 2

	def step_neglecting_vertical_motion(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_stepping_by_neglecting_vertical_motion is None:
			self._stencils_stepping_by_neglecting_vertical_motion_initialize(
				raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# Run the compute function of the stencil stepping the solution
		self._stencil_stepping_by_neglecting_vertical_motion.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated x-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])

		# Bring the updated y-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		# Bring the updated water constituents back to the original dimensions,
		# and enforce the boundary conditions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def step_integrating_vertical_advection(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time by integrating the vertical advection, i.e.,
		by accounting for the change over time in potential temperature.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.
		"""
		raise NotImplementedError()

	def step_integrating_sedimentation_flux(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the mass fraction of precipitation water by taking
		the sedimentation into account. For the sake of numerical stability,
		a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a time step which may be smaller than that specified by the user.
		"""
		raise NotImplementedError()

	def _stencils_stepping_by_neglecting_vertical_motion_initialize(
		self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing the RK2 scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_outputs()

		# Set the stencils' computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_mtg_int': self._in_mtg,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Instantiate the stencil
		self._stencil_stepping_by_neglecting_vertical_motion = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Call parent method
		super()._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Determine the size of the arrays which will serve as stencils'
		# inputs and outputs. These arrays may be shared with the stencil
		# in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((li, lj, nz), dtype=dtype)
		self._in_su_int = np.zeros((li, lj, nz), dtype=dtype)
		self._in_sv_int = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_int = np.zeros((li, lj, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((li, lj, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((li, lj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_set_inputs(
		self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			s_tnd_on  = raw_tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = raw_tendencies.get(mf_wv, None) is not None
			qc_tnd_on = raw_tendencies.get(mf_clw, None) is not None
			qr_tnd_on = raw_tendencies.get(mf_pw, None) is not None
			su_tnd_on = raw_tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = raw_tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = (1./2. + 1./2.*(stage > 0)) * dt.total_seconds()

		if stage == 0:
			# Extract the Numpy arrays representing the current solution
			s	= raw_state['air_isentropic_density']
			su	= raw_state['x_momentum_isentropic']
			sv	= raw_state['y_momentum_isentropic']
			if self._moist_on:
				sqv = raw_state['isentropic_density_of_water_vapor']
				sqc = raw_state['isentropic_density_of_cloud_liquid_water']
				sqr = raw_state['isentropic_density_of_precipitation_water']

			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_su[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the intermediate solution
		s	= raw_state['air_isentropic_density']
		u	= raw_state['x_velocity_at_u_locations']
		v	= raw_state['y_velocity_at_v_locations']
		mtg = raw_state['montgomery_potential']
		su	= raw_state['x_momentum_isentropic']
		sv	= raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']

		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(s)
		self._in_u[:mi+1, :mj, :] = self._hboundary.from_physical_to_computational_domain(u)
		self._in_v[:mi, :mj+1, :] = self._hboundary.from_physical_to_computational_domain(v)
		self._in_mtg[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(mtg)
		self._in_su_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(su)
		self._in_sv_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sv)
		if self._moist_on:
			self._in_sqv_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqv)
			self._in_sqc_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqc)
			self._in_sqr_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if s_tnd_on:
			s_tnd = raw_tendencies['air_isentropic_density']
			self._in_s_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s_tnd)
		if qv_tnd_on:
			qv_tnd = raw_tendencies[mf_wv]
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			qc_tnd = raw_tendencies[mf_clw]
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			qr_tnd = raw_tendencies[mf_pw]
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)
		if su_tnd_on:
			su_tnd = raw_tendencies['x_momentum_isentropic']
			self._in_su_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su_tnd)
		if sv_tnd_on:
			sv_tnd = raw_tendencies['y_momentum_isentropic']
			self._in_sv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv_tnd)

	def _stencil_stepping_by_neglecting_vertical_motion_defs(
		self, dt, in_s, in_s_int, in_u_int, in_v_int, in_mtg_int,
		in_su, in_su_int, in_sv, in_sv_int,
		in_sqv=None, in_sqv_int=None,
		in_sqc=None, in_sqc_int=None,
		in_sqr=None, in_sqr_int=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via a stage of the RK2 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, s_tnd=in_s_tnd, su_tnd=in_su_tnd,
							sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Calculate the pressure gradient
		pgx, pgy = self._get_pressure_gradient(i, j, in_mtg_int)

		# Advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
											 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
											 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
											 in_s_tnd[i, j])

		# Advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
											   in_s_int[i, j] * pgx[i, j])
		else:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
											   in_s_int[i, j] * pgx[i, j] - in_su_tnd[i, j])

		# Advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
											   in_s_int[i, j] * pgy[i, j])
		else:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
											   in_s_int[i, j] * pgy[i, j] - in_sv_tnd[i, j])

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr

	def _get_pressure_gradient(self, i, j, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Output fields
		pgx = gt.Equation()
		pgy = gt.Equation()

		# Computations
		if self._hflux_scheme in ['fifth_order_upwind']:
			pgx[i, j] = (in_mtg[i-2, j] - 8. * in_mtg[i-1, j] +
						 8. * in_mtg[i+1, j] - in_mtg[i+2, j]) / (12. * dx)
			pgy[i, j] = (in_mtg[i, j-2] - 8. * in_mtg[i, j-1] +
						 8. * in_mtg[i, j+1] - in_mtg[i, j+2]) / (12. * dy)
		else:
			pgx[i, j] = (in_mtg[i+1, j] - in_mtg[i-1, j]) / (2. * dx)
			pgy[i, j] = (in_mtg[i, j+1] - in_mtg[i, j-1]) / (2. * dy)

		return pgx, pgy


class RK3COSMO(RK2):
	"""
	This class inherits
	:class:`~tasmania.dynamics._prognostic_isentropic._RK2`
	to implement the three-stages, second-order Runge-Kutta scheme which takes over the
	prognostic part of the three-dimensional moist isentropic dynamical core.
	This is the time-marching scheme used in the COSMO model.
	"""
	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, diagnostics, horizontal_boundary_conditions,
						 horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
						 sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
						 raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 3

	def _stencils_stepping_by_neglecting_vertical_motion_set_inputs(
		self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Call the parent's method
		super()._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# Update the local time step
		self._dt.value = (1./3. + 1./6.*(stage > 0) + 1./2.*(stage > 1)) * dt.total_seconds()

	def _get_pressure_gradient(self, i, j, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Output fields
		pgx = gt.Equation()
		pgy = gt.Equation()

		# Computations
		if self._hflux_scheme in ['third_order_upwind', 'fifth_order_upwind']:
			pgx[i, j] = (in_mtg[i-2, j] - 8. * in_mtg[i-1, j] +
						 8. * in_mtg[i+1, j] - in_mtg[i+2, j]) / (12. * dx)
			pgy[i, j] = (in_mtg[i, j-2] - 8. * in_mtg[i, j-1] +
						 8. * in_mtg[i, j+1] - in_mtg[i, j+2]) / (12. * dy)
		else:
			pgx[i, j] = (in_mtg[i+1, j] - in_mtg[i-1, j]) / dx
			pgy[i, j] = (in_mtg[i, j+1] - in_mtg[i, j-1]) / dy

		return pgx, pgy


class RK3(IsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.prognostic_isentropic.PrognosticIsentropic`
	to implement the three-stages, third-order Runge-Kutta scheme which takes over the
	prognostic part of the three-dimensional moist isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, diagnostics,
				 horizontal_boundary_conditions, horizontal_flux_scheme,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_scheme=None,
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, diagnostics, horizontal_boundary_conditions,
						 horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
						 sedimentation_on, sedimentation_flux_scheme, sedimentation_substeps,
						 raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be re-directed when the corresponding forward methods
		# are invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_motion_first = None
		self._stencil_stepping_by_neglecting_vertical_motion_second = None
		self._stencil_stepping_by_neglecting_vertical_motion_third = None
		self._stencil_computing_slow_tendencies = None
		self._stencil_ensuring_vertical_cfl_is_obeyed = None
		self._stencil_stepping_by_integrating_sedimentation_flux = None

		# Free parameters for RK3
		self._alpha1 = 1./2.
		self._alpha2 = 3./4.

		# Set the other parameters yielding a third-order method
		self._gamma1 = (3.*self._alpha2 - 2.) / \
					   (6. * self._alpha1 * (self._alpha2 - self._alpha1))
		self._gamma2 = (3.*self._alpha1 - 2.) / \
					   (6. * self._alpha2 * (self._alpha1 - self._alpha2))
		self._gamma0 = 1. - self._gamma1 - self._gamma2
		self._beta21 = self._alpha2 - 1. / (6. * self._alpha1 * self._gamma2)

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 3

	def step_neglecting_vertical_motion(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_stepping_by_neglecting_vertical_motion_first is None:
			self._stencils_stepping_by_neglecting_vertical_motion_initialize(
				raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencils_stepping_by_neglecting_vertical_motion_set_inputs(
			stage, dt, raw_state, raw_tendencies)

		# Run the compute function of the stencil stepping the solution
		self._stencils_stepping_by_neglecting_vertical_motion_compute(stage)

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated x-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])

		# Bring the updated y-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		# Bring the updated water constituents back to the original dimensions,
		# and enforce the boundary conditions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def step_integrating_vertical_advection(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time by integrating the vertical advection, i.e.,
		by accounting for the change over time in potential temperature.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.
		"""
		raise NotImplementedError()

	def step_integrating_sedimentation_flux(self, stage, dt, raw_state_now, raw_state_prv):
		"""
		Method advancing the mass fraction of precipitation water by taking
		the sedimentation into account. For the sake of numerical stability,
		a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a time step which may be smaller than that specified by the user.
		"""
		raise NotImplementedError()

	def _stencils_stepping_by_neglecting_vertical_motion_initialize(
		self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing the RK2 scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store temporary fields to be
		# shared across the different stencils which step the solution
		# disregarding any vertical motion
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_temporaries()

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_motion_allocate_outputs()

		# Set the stencils' computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the first stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_mtg': self._in_mtg, 'in_su': self._in_su, 'in_sv': self._in_sv}
		_outputs = {'out_s0': self._s0, 'out_s': self._out_s,
					'out_su0': self._su0, 'out_su': self._out_su,
					'out_sv0': self._sv0, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int})
			_outputs.update({'out_sqv0': self._sqv0, 'out_sqv': self._out_sqv,
							 'out_sqc0': self._sqc0, 'out_sqc': self._out_sqc,
							 'out_sqr0': self._sqr0, 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Instantiate the first stencil
		self._stencil_stepping_by_neglecting_vertical_motion_first = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_first_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

		# Set the second stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_mtg_int': self._in_mtg,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
				   'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0}
		_outputs = {'out_s1': self._s1, 'out_s': self._out_s,
					'out_su1': self._su1, 'out_su': self._out_su,
					'out_sv1': self._sv1, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
							'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
							'in_sqr0': self._sqr0})
			_outputs.update({'out_sqv1': self._sqv1, 'out_sqv': self._out_sqv,
							 'out_sqc1': self._sqc1, 'out_sqc': self._out_sqc,
							 'out_sqr1': self._sqr1, 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Instantiate the second stencil
		self._stencil_stepping_by_neglecting_vertical_motion_second = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_second_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

		# Set the third stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_mtg_int': self._in_mtg,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
				   'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0,
				   'in_s1': self._s1, 'in_su1': self._su1, 'in_sv1': self._sv1}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
							'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
							'in_sqr0': self._sqr0, 'in_sqv1': self._sqv1,
							'in_sqc1': self._sqc1, 'in_sqr1': self._sqr1})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd
			if raw_tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if raw_tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd

		# Instantiate the third stencil
		self._stencil_stepping_by_neglecting_vertical_motion_third = gt.NGStencil(
			definitions_func = self._stencil_stepping_by_neglecting_vertical_motion_third_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Call parent method
		super()._stencils_stepping_by_neglecting_vertical_motion_allocate_inputs(raw_tendencies)

		# Determine the size of the arrays which will serve as stencils'
		# inputs and outputs. These arrays may be shared with the stencil
		# in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((li, lj, nz), dtype=dtype)
		self._in_su_int = np.zeros((li, lj, nz), dtype=dtype)
		self._in_sv_int = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_int = np.zeros((li, lj, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((li, lj, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((li, lj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_allocate_temporaries(self):
		"""
		Allocate the Numpy arrays which store temporary fields to be shared across
		the different GT4Py stencils which step the solution disregarding any
		vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Determine the size of the arrays which will serve as stencils'
		# inputs and outputs. These arrays may be shared with the stencil
		# in charge of integrating the vertical advection
		li = mi if self._adiabatic_flow else max(mi, nx)
		lj = mj if self._adiabatic_flow else max(mj, ny)

		# Allocate the Numpy arrays which will store the first increment
		# for all prognostic variables
		self._s0  = np.zeros((li, lj, nz), dtype=dtype)
		self._su0 = np.zeros((li, lj, nz), dtype=dtype)
		self._sv0 = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._sqv0 = np.zeros((li, lj, nz), dtype=dtype)
			self._sqc0 = np.zeros((li, lj, nz), dtype=dtype)
			self._sqr0 = np.zeros((li, lj, nz), dtype=dtype)

		# Allocate the Numpy arrays which will store the second increment
		# for all prognostic variables
		self._s1  = np.zeros((li, lj, nz), dtype=dtype)
		self._su1 = np.zeros((li, lj, nz), dtype=dtype)
		self._sv1 = np.zeros((li, lj, nz), dtype=dtype)
		if self._moist_on:
			self._sqv1 = np.zeros((li, lj, nz), dtype=dtype)
			self._sqc1 = np.zeros((li, lj, nz), dtype=dtype)
			self._sqr1 = np.zeros((li, lj, nz), dtype=dtype)

	def _stencils_stepping_by_neglecting_vertical_motion_set_inputs(
		self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			s_tnd_on  = raw_tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = raw_tendencies.get(mf_wv, None) is not None
			qc_tnd_on = raw_tendencies.get(mf_clw, None) is not None
			qr_tnd_on = raw_tendencies.get(mf_pw, None) is not None
			su_tnd_on = raw_tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = raw_tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = dt.total_seconds()

		# Extract the Numpy arrays representing the intermediate solution
		s	= raw_state['air_isentropic_density']
		u	= raw_state['x_velocity_at_u_locations']
		v	= raw_state['y_velocity_at_v_locations']
		mtg = raw_state['montgomery_potential']
		su	= raw_state['x_momentum_isentropic']
		sv	= raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']

		if stage == 0:
			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_u[:mi+1, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u)
			self._in_v[:mi, :mj+1, :] = \
				self._hboundary.from_physical_to_computational_domain(v)
			self._in_mtg[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(mtg)
			self._in_su[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)
		else:

			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_u[:mi+1, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u)
			self._in_v[:mi, :mj+1, :] = \
				self._hboundary.from_physical_to_computational_domain(v)
			self._in_mtg[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(mtg)
			self._in_su_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if s_tnd_on:
			s_tnd = raw_tendencies['air_isentropic_density']
			self._in_s_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s_tnd)
		if qv_tnd_on:
			qv_tnd = raw_tendencies[mf_wv]
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			qc_tnd = raw_tendencies[mf_clw]
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			qr_tnd = raw_tendencies[mf_pw]
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)
		if su_tnd_on:
			su_tnd = raw_tendencies['x_momentum_isentropic']
			self._in_su_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su_tnd)
		if sv_tnd_on:
			sv_tnd = raw_tendencies['y_momentum_isentropic']
			self._in_sv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv_tnd)

	def _stencils_stepping_by_neglecting_vertical_motion_compute(self, stage):
		if stage == 0:
			self._stencil_stepping_by_neglecting_vertical_motion_first.compute()
		elif stage == 1:
			self._stencil_stepping_by_neglecting_vertical_motion_second.compute()
		else:
			self._stencil_stepping_by_neglecting_vertical_motion_third.compute()

	def _stencil_stepping_by_neglecting_vertical_motion_first_defs(
		self, dt, in_s, in_u, in_v, in_mtg,	in_su, in_sv,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the first stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		a1 = self._alpha1

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		out_s0  = gt.Equation()
		out_su0 = gt.Equation()
		out_sv0 = gt.Equation()
		if self._moist_on:
			out_sqv0 = gt.Equation()
			out_sqc0 = gt.Equation()
			out_sqr0 = gt.Equation()

		# Instantiate fields representing the output state
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_su, in_sv,
							s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_mtg,
							in_su, in_sv, in_sqv, in_sqc, in_sqr,
							in_s_tnd, in_su_tnd, in_sv_tnd,
							in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Calculate the pressure gradient
		pgx, pgy = self._get_pressure_gradient(i, j, in_mtg)

		# Advance the isentropic density
		if in_s_tnd is None:
			out_s0[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
							   	   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			out_s0[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
								   in_s_tnd[i, j])
		out_s[i, j] = in_s[i, j] + a1 * out_s0[i, j]

		# Advance the x-momentum
		if in_su_tnd is None:
			out_su0[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s[i, j] * pgx[i, j])
		else:
			out_su0[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s[i, j] * pgx[i, j] - in_su_tnd[i, j])
		out_su[i, j] = in_su[i, j] + a1 * out_su0[i, j]

		# Advance the y-momentum
		if in_sv_tnd is None:
			out_sv0[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
							  	    (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s[i, j] * pgy[i, j])
		else:
			out_sv0[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s[i, j] * pgy[i, j] - in_sv_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + a1 * out_sv0[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv0[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  	 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv0[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  	 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
									  	 in_s[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + a1 * out_sqv0[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc0[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  	 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc0[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  	 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
									  	 in_s[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + a1 * out_sqc0[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr0[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  	 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr0[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  	 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
									  	 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + a1 * out_sqr0[i, j]

		if not self._moist_on:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv
		else:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv, \
				   out_sqv0, out_sqv, out_sqc0, out_sqc, out_sqr0, out_sqr

	def _stencil_stepping_by_neglecting_vertical_motion_second_defs(
		self, dt, in_s, in_s_int, in_s0, in_u_int, in_v_int, in_mtg_int,
		in_su, in_su_int, in_su0, in_sv, in_sv_int, in_sv0,
		in_sqv=None, in_sqv_int=None, in_sqv0=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the second stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		a2, b21 = self._alpha2, self._beta21

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		out_s1  = gt.Equation()
		out_su1 = gt.Equation()
		out_sv1 = gt.Equation()
		if self._moist_on:
			out_sqv1 = gt.Equation()
			out_sqc1 = gt.Equation()
			out_sqr1 = gt.Equation()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, s_tnd=in_s_tnd, su_tnd=in_su_tnd,
							sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Calculate the pressure gradient
		pgx, pgy = self._get_pressure_gradient(i, j, in_mtg_int)

		# Advance the isentropic density
		if in_s_tnd is None:
			out_s1[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			out_s1[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
								   in_s_tnd[i, j])
		out_s[i, j] = in_s[i, j] + b21 * in_s0[i, j] + (a2 - b21) * out_s1[i, j]

		# Advance the x-momentum
		if in_su_tnd is None:
			out_su1[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s_int[i, j] * pgx[i, j])
		else:
			out_su1[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s_int[i, j] * pgx[i, j] - in_su_tnd[i, j])
		out_su[i, j] = in_su[i, j] + b21 * in_su0[i, j] + (a2 - b21) * out_su1[i, j]

		# Advance the y-momentum
		if in_sv_tnd is None:
			out_sv1[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s_int[i, j] * pgy[i, j])
		else:
			out_sv1[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s_int[i, j] * pgy[i, j] - in_sv_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + b21 * in_sv0[i, j] + (a2 - b21) * out_sv1[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv1[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv1[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + b21 * in_sqv0[i, j] + (a2 - b21) * out_sqv1[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc1[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc1[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + b21 * in_sqc0[i, j] + (a2 - b21) * out_sqc1[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr1[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr1[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
										 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + b21 * in_sqr0[i, j] + (a2 - b21) * out_sqr1[i, j]

		if not self._moist_on:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv
		else:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv, \
				   out_sqv1, out_sqv, out_sqc1, out_sqc, out_sqr1, out_sqr

	def _stencil_stepping_by_neglecting_vertical_motion_third_defs(
		self, dt, in_s, in_s_int, in_s0, in_s1, in_u_int, in_v_int, in_mtg_int,
		in_su, in_su_int, in_su0, in_su1, in_sv, in_sv_int, in_sv0, in_sv1,
		in_sqv=None, in_sqv_int=None, in_sqv0=None, in_sqv1=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None, in_sqc1=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None, in_sqr1=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the second stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		tmp_s2 = gt.Equation()
		tmp_su2 = gt.Equation()
		tmp_sv2 = gt.Equation()
		if self._moist_on:
			tmp_sqv2 = gt.Equation()
			tmp_sqc2 = gt.Equation()
			tmp_sqr2 = gt.Equation()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, s_tnd=in_s_tnd, su_tnd=in_su_tnd,
							sv_tnd=in_sv_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int, in_mtg_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Calculate the pressure gradient
		pgx, pgy = self._get_pressure_gradient(i, j, in_mtg_int)

		# Advance the isentropic density
		if in_s_tnd is None:
			tmp_s2[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		else:
			tmp_s2[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
								   in_s_tnd[i, j])
		out_s[i, j] = in_s[i, j] + g0 * in_s0[i, j] \
					  			 + g1 * in_s1[i, j] \
					  			 + g2 * tmp_s2[i, j]

		# Advance the x-momentum
		if in_su_tnd is None:
			tmp_su2[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s_int[i, j] * pgx[i, j])
		else:
			tmp_su2[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
									in_s_int[i, j] * pgx[i, j] - in_su_tnd[i, j])
		out_su[i, j] = in_su[i, j] + g0 * in_su0[i, j] \
					   			   + g1 * in_su1[i, j] \
					   			   + g2 * tmp_su2[i, j]

		# Advance the y-momentum
		if in_sv_tnd is None:
			tmp_sv2[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s_int[i, j] * pgy[i, j])
		else:
			tmp_sv2[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
									in_s_int[i, j] * pgy[i, j] - in_sv_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + g0 * in_sv0[i, j] \
					   			   + g1 * in_sv1[i, j] \
					   			   + g2 * tmp_sv2[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				tmp_sqv2[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				tmp_sqv2[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + g0 * in_sqv0[i, j] \
										 + g1 * in_sqv1[i, j] \
										 + g2 * tmp_sqv2[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				tmp_sqc2[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				tmp_sqc2[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + g0 * in_sqc0[i, j] \
										 + g1 * in_sqc1[i, j] \
										 + g2 * tmp_sqc2[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				tmp_sqr2[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				tmp_sqr2[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
										 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + g0 * in_sqr0[i, j] \
										 + g1 * in_sqr1[i, j] \
										 + g2 * tmp_sqr2[i, j]

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr

	def _get_pressure_gradient(self, i, j, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Output fields
		pgx = gt.Equation()
		pgy = gt.Equation()

		# Computations
		if self._hflux_scheme in ['third_order_upwind', 'fifth_order_upwind']:
			pgx[i, j] = (in_mtg[i-2, j] - 8. * in_mtg[i-1, j] +
						 8. * in_mtg[i+1, j] - in_mtg[i+2, j]) / (12. * dx)
			pgy[i, j] = (in_mtg[i, j-2] - 8. * in_mtg[i, j-1] +
						 8. * in_mtg[i, j+1] - in_mtg[i, j+2]) / (12. * dy)
		else:
			pgx[i, j] = (in_mtg[i+1, j] - in_mtg[i-1, j]) / (2. * dx)
			pgy[i, j] = (in_mtg[i, j+1] - in_mtg[i, j-1]) / (2. * dy)

		return pgx, pgy

