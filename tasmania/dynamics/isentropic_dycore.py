"""
This module contains:
	IsentropicDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.dynamics.diagnostics import HorizontalVelocity, \
										  IsentropicDiagnostics, \
										  WaterConstituent
from tasmania.core.dycore import DynamicalCore
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics.vertical_damping import VerticalDamping

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class IsentropicDynamicalCore(DynamicalCore):
	"""
	This class inherits :class:`~tasmania.dynamics.dycore.DynamicalCore`
	to implement the three-dimensional (moist) isentropic dynamical core.
	The class supports different numerical schemes to carry out the prognostic
	steps of the dynamical core, and different types of lateral boundary conditions.
	The conservative form of the governing equations is used.
	"""
	def __init__(self, grid, moist_on, time_integration_scheme,
				 horizontal_flux_scheme, horizontal_boundary_type,
				 intermediate_parameterizations=None,
				 diagnostics=None,
				 damp_on=True, damp_type='rayleigh', damp_depth=15,
				 damp_max=0.0002, damp_at_every_stage=True,
				 smooth_on=True, smooth_type='first_order', smooth_damp_depth=10,
				 smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
				 smooth_moist_on=False, smooth_moist_type='first_order',
				 smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
				 smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_type='first_order_upwind',
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		time_integration_scheme : str
			String specifying the time stepping method to implement.
			See :class:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic`
			for all available options.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalIsentropicFlux`
			for all available options.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		intermediate_parameterizations : `obj`, None
			:class:`~tasmania.physics.composite.PhysicsComponentComposite`
			object, wrapping the fast physical parameterizations.
			Here, *fast* refers to the fact that these parameterizations
			are evaluated *before* each stage of the dynamical core.
			In essence, feeding the dynamical core with fast parameterizations
			allows to pursue the concurrent splitting strategy.
		diagnostics : `obj`, None
			:class:`~tasmania.core.physics_composite.DiagnosticComponentComposite`
			object  wrapping a set of diagnostic parameterizations, evaluated
			at the end of each stage of the dynamical core.
		damp_on : `bool`, optional
			:obj:`True` to enable vertical damping, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. Defaults to 'rayleigh'.
			See :class:`~tasmania.dynamics.vertical_damping.VerticalDamping`
			for all available options.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		damp_at_every_stage : `bool`, optional
			:obj:`True` to carry out the damping at each stage performed by the
			dynamical core, :obj:`False` to carry out the damping only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_on : `bool`, optional
			:obj:`True` to enable numerical smoothing, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement.
			Defaults to 'first-order'. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for all available options.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Defaults to 10.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Defaults to 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. Defaults to 0.24.
			See :class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for further details.
		smooth_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing at each stage performed by the
			dynamical core, :obj:`False` to apply numerical smoothing only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_moist_on : `bool`, optional
			:obj:`True` to enable numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply to the water constituents.
			Defaults to 'first-order'. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for all available options.
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			Defaults to 0.24. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for further details.
		smooth_moist_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing to the water constituents
			at each stage performed by the dynamical core, :obj:`False` to apply
			numerical smoothing only at the end of each timestep. Defaults to :obj:`True`.
		adiabatic_flow : `bool`, optional
			:obj:`True` for an adiabatic atmosphere, in which the potential temperature
			is conserved, :obj:`False` otherwise. Defaults to :obj:`True`.
		sedimentation_on : `bool`, optional
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		sedimentation_flux_type : `str`, optional
			String specifying the method used to compute the numerical sedimentation flux.
			See :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
			for all available options.
		sedimentation_substeps : `int`, optional
			Number of sub-timesteps to perform in order to integrate the sedimentation flux.
			Defaults to 2.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils
			implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. See
			:class:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic`
			and	:class:`~tasmania.dynamics.diagnostics.IsentropicDiagnostics`
			for the physical constants used.
		"""
		# Keep track of the input parameters
		self._damp_on                      = damp_on
		self._damp_at_every_stage		   = damp_at_every_stage
		self._smooth_on                    = smooth_on
		self._smooth_at_every_stage		   = smooth_at_every_stage
		self._smooth_moist_on              = smooth_moist_on
		self._smooth_moist_at_every_stage  = smooth_moist_at_every_stage
		self._adiabatic_flow 			   = adiabatic_flow
		self._sedimentation_on             = sedimentation_on
		self._dtype						   = dtype

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = IsentropicDiagnostics(grid, backend, dtype, physical_constants)

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid)

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = IsentropicPrognostic.factory(
			time_integration_scheme, grid, moist_on, self._diagnostic, self._boundary,
			horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
			sedimentation_on, sedimentation_flux_type, sedimentation_substeps,
			raindrop_fall_velocity_diagnostic, backend, dtype, physical_constants)

		# Instantiate the class in charge of applying vertical damping
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if damp_on: 
			self._damper = VerticalDamping.factory(damp_type, (nx, ny, nz), grid,
												   damp_depth, damp_max, backend, dtype)

		# Instantiate the classes in charge of applying numerical smoothing
		if smooth_on:
			self._smoother = HorizontalSmoothing.factory(smooth_type, (nx, ny, nz), grid,
														 smooth_damp_depth, smooth_coeff,
														 smooth_coeff_max, backend, dtype)
			if moist_on and smooth_moist_on:
				self._smoother_moist = HorizontalSmoothing.factory(
					smooth_moist_type, (nx, ny, nz), grid, smooth_moist_damp_depth,
					smooth_moist_coeff, smooth_moist_coeff_max, backend, dtype)

		# Instantiate the class in charge of diagnosing the velocity components
		self._velocity_components = HorizontalVelocity(grid, backend, dtype)

		# Instantiate the class in charge of diagnosing the mass fraction and
		# isentropic density of the water constituents
		if moist_on:
			self._water_constituent = WaterConstituent(grid, backend, dtype)

		# Set the pointer to the private method implementing each stage
		self._array_call = self._array_call_dry if not moist_on else self._array_call_moist

		# Call parent constructor
		super().__init__(grid, moist_on, intermediate_parameterizations, diagnostics)

	@property
	def _input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0],
					  self._grid.z.dims[0])
		dims_stg_y = (self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0],
					  self._grid.z.dims[0])
		dims_stg_z = (self._grid.x.dims[0], self._grid.y.dims[0],
					  self._grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'air_pressure_on_interface_levels': {'dims': dims_stg_z, 'units': 'Pa'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

			if self._sedimentation_on:
				return_dict['accumulated_precipitation'] = {'dims': dims, 'units': 'mm'}

		return return_dict

	@property
	def _tendency_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def _output_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0],
					  self._grid.z.dims[0])
		dims_stg_y = (self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0],
					  self._grid.z.dims[0])
		dims_stg_z = (self._grid.x.dims[0], self._grid.y.dims[0],
					  self._grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'air_pressure_on_interface_levels': {'dims': dims_stg_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_stg_z, 'units': 'm^2 s^-2 K^-1'},
			'height_on_interface_levels': {'dims': dims_stg_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._moist_on:
			return_dict['air_density'] = {'dims': dims, 'units': 'kg m^-3'}
			return_dict['air_temperature'] = {'dims': dims, 'units': 'K'}
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

			if self._sedimentation_on:
				return_dict['accumulated_precipitation'] = {'dims': dims, 'units': 'mm'}
				return_dict['precipitation'] = {'dims': dims, 'units': 'mm h^-1'}

		return return_dict

	@property
	def stages(self):
		return self._prognostic.stages

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the isentropic dynamical core, either dry or moist.
		"""
		return self._array_call(stage, raw_state, raw_tendencies, timestep)

	def _array_call_dry(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the dry dynamical core.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Perform the prognostic step
		raw_state_new = self._prognostic.step_neglecting_vertical_motion(
			stage, timestep, raw_state, raw_tendencies)

		damped = False
		if self._damp_on:
			# If this is the first call to the entry-point method,
			# set the reference state...
			if not hasattr(self, '_s_ref'):
				self._s_ref  = np.copy(raw_state['air_isentropic_density'])
				self._su_ref = np.copy(raw_state['x_momentum_isentropic'])
				self._sv_ref = np.copy(raw_state['y_momentum_isentropic'])

			# ...and allocate memory to store damped fields
			if not hasattr(self, '_s_damped'):
				self._s_damped  = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)

			if self._damp_at_every_stage or stage == self.stages-1:
				damped = True

				# Extract the current prognostic model variables
				s_now_  = raw_state['air_isentropic_density']
				su_now_ = raw_state['x_momentum_isentropic']
				sv_now_ = raw_state['y_momentum_isentropic']

				# Extract the stepped prognostic model variables
				s_new_  = raw_state_new['air_isentropic_density']
				su_new_ = raw_state_new['x_momentum_isentropic']
				sv_new_ = raw_state_new['y_momentum_isentropic']

				# Apply vertical damping
				self._damper(timestep, s_now_,  s_new_,  self._s_ref,  self._s_damped)
				self._damper(timestep, su_now_, su_new_, self._su_ref, self._su_damped)
				self._damper(timestep, sv_now_, sv_new_, self._sv_ref, self._sv_damped)

		# Properly set pointers to current solution
		s_new  = self._s_damped if damped else raw_state_new['air_isentropic_density']
		su_new = self._su_damped if damped else raw_state_new['x_momentum_isentropic']
		sv_new = self._sv_damped if damped else raw_state_new['y_momentum_isentropic']

		smoothed = False
		if self._smooth_on:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_s_smoothed'):
				self._s_smoothed  = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_at_every_stage or stage == self.stages-1:
				smoothed = True

				# Apply horizontal smoothing
				self._smoother(s_new,  self._s_smoothed)
				self._smoother(su_new, self._su_smoothed)
				self._smoother(sv_new, self._sv_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._s_smoothed,  s_new)
				self._boundary.enforce(self._su_smoothed, su_new)
				self._boundary.enforce(self._sv_smoothed, sv_new)

		# Properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		# Diagnose the velocity components
		u_out, v_out = self._velocity_components.get_velocity_components(s_out, su_out, sv_out)
		self._boundary.set_outermost_layers_x(u_out, raw_state['x_velocity_at_u_locations'])
		self._boundary.set_outermost_layers_y(v_out, raw_state['y_velocity_at_v_locations'])

		# Diagnose the pressure, the Exner function, the Montgomery potential
		# and the geometric height of the half levels
		pt = raw_state['air_pressure_on_interface_levels'][0, 0, 0]
		p_out, exn_out, mtg_out, h_out = self._diagnostic.get_diagnostic_variables(s_out, pt)

		# Instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out,
			'air_pressure_on_interface_levels': p_out,
			'exner_function_on_interface_levels': exn_out,
			'height_on_interface_levels': h_out,
			'montgomery_potential': mtg_out,
			'x_momentum_isentropic': su_out,
			'x_velocity_at_u_locations': u_out,
			'y_momentum_isentropic': sv_out,
			'y_velocity_at_v_locations': v_out,
		}

		return raw_state_out

	def _array_call_moist(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the dry dynamical core.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Allocate memory to store the isentropic density of all water constituents
		if not hasattr(self, '_sqv_now'):
			self._sqv_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr_now = np.zeros((nx, ny, nz), dtype=dtype)

		# Diagnosed the isentropic density of all water constituents
		s_now  = raw_state['air_isentropic_density']
		qv_now = raw_state['mass_fraction_of_water_vapor_in_air']
		qc_now = raw_state['mass_fraction_of_cloud_liquid_water_in_air']
		qr_now = raw_state['mass_fraction_of_precipitation_water_in_air']
		self._water_constituent.get_density_of_water_constituent(s_now, qv_now, self._sqv_now)
		self._water_constituent.get_density_of_water_constituent(s_now, qc_now, self._sqc_now)
		self._water_constituent.get_density_of_water_constituent(s_now, qr_now, self._sqr_now)
		raw_state['isentropic_density_of_water_vapor']         = self._sqv_now
		raw_state['isentropic_density_of_cloud_liquid_water']  = self._sqc_now
		raw_state['isentropic_density_of_precipitation_water'] = self._sqr_now

		# Perform the prognostic step
		raw_state_new = self._prognostic.step_neglecting_vertical_motion(
			stage, timestep, raw_state, raw_tendencies)

		damped = False
		if self._damp_on:
			# If this is the first call to the entry-point method,
			# set the reference state...
			if not hasattr(self, '_s_ref'):
				self._s_ref   = np.copy(raw_state['air_isentropic_density'])
				self._su_ref  = np.copy(raw_state['x_momentum_isentropic'])
				self._sv_ref  = np.copy(raw_state['y_momentum_isentropic'])
				self._sqv_ref = np.copy(raw_state['isentropic_density_of_water_vapor'])
				self._sqc_ref = np.copy(raw_state['isentropic_density_of_cloud_liquid_water'])
				self._sqr_ref = np.copy(raw_state['isentropic_density_of_precipitation_water'])

			# ...and allocate memory to store damped fields
			if not hasattr(self, '_s_damped'):
				self._s_damped   = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_damped  = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_damped  = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqv_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqc_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqr_damped = np.zeros((nx, ny, nz), dtype=dtype)

			if self._damp_at_every_stage or stage == self.stages-1:
				damped = True

				# Extract the current prognostic model variables
				s_now_   = raw_state['air_isentropic_density']
				su_now_  = raw_state['x_momentum_isentropic']
				sv_now_  = raw_state['y_momentum_isentropic']
				sqv_now_ = raw_state['isentropic_density_of_water_vapor']
				sqc_now_ = raw_state['isentropic_density_of_cloud_liquid_water']
				sqr_now_ = raw_state['isentropic_density_of_precipitation_water']

				# Extract the stepped prognostic model variables
				s_new_   = raw_state_new['air_isentropic_density']
				su_new_  = raw_state_new['x_momentum_isentropic']
				sv_new_  = raw_state_new['y_momentum_isentropic']
				sqv_new_ = raw_state_new['isentropic_density_of_water_vapor']
				sqc_new_ = raw_state_new['isentropic_density_of_cloud_liquid_water']
				sqr_new_ = raw_state_new['isentropic_density_of_precipitation_water']

				# Apply vertical damping
				self._damper(timestep, s_now_,   s_new_,   self._s_ref,   self._s_damped)
				self._damper(timestep, su_now_,  su_new_,  self._su_ref,  self._su_damped)
				self._damper(timestep, sv_now_,  sv_new_,  self._sv_ref,  self._sv_damped)
				self._damper(timestep, sqv_now_, sqv_new_, self._sqv_ref, self._sqv_damped)
				self._damper(timestep, sqc_now_, sqc_new_, self._sqc_ref, self._sqc_damped)
				self._damper(timestep, sqr_now_, sqr_new_, self._sqr_ref, self._sqr_damped)

		# Properly set pointers to current solution
		s_new   = self._s_damped if damped else raw_state_new['air_isentropic_density']
		su_new  = self._su_damped if damped else raw_state_new['x_momentum_isentropic']
		sv_new  = self._sv_damped if damped else raw_state_new['y_momentum_isentropic']
		sqv_new = self._sqv_damped if damped else \
				  raw_state_new['isentropic_density_of_water_vapor']
		sqc_new = self._sqc_damped if damped else \
				  raw_state_new['isentropic_density_of_cloud_liquid_water']
		sqr_new = self._sqr_damped if damped else \
				  raw_state_new['isentropic_density_of_precipitation_water']

		smoothed = False
		if self._smooth_on:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_s_smoothed'):
				self._s_smoothed  = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_at_every_stage or stage == self.stages-1:
				smoothed = True

				# Apply horizontal smoothing
				self._smoother(s_new,  self._s_smoothed)
				self._smoother(su_new, self._su_smoothed)
				self._smoother(sv_new, self._sv_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._s_smoothed,  s_new)
				self._boundary.enforce(self._su_smoothed, su_new)
				self._boundary.enforce(self._sv_smoothed, sv_new)

		# Properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		smoothed_moist = False
		if self._smooth_moist_on:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_sqv_smoothed'):
				self._sqv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqc_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqr_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_moist_at_every_stage or stage == self.stages-1:
				smoothed_moist = True

				# Apply horizontal smoothing
				self._smoother(sqv_new, self._sqv_smoothed)
				self._smoother(sqc_new, self._sqc_smoothed)
				self._smoother(sqr_new, self._sqr_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._sqv_smoothed, sqv_new)
				self._boundary.enforce(self._sqc_smoothed, sqc_new)
				self._boundary.enforce(self._sqr_smoothed, sqr_new)

		# Properly set pointers to output solution
		sqv_out = self._sqv_smoothed if smoothed_moist else sqv_new
		sqc_out = self._sqc_smoothed if smoothed_moist else sqc_new
		sqr_out = self._sqr_smoothed if smoothed_moist else sqr_new

		# Diagnose the velocity components
		u_out, v_out = self._velocity_components.get_velocity_components(s_out, su_out, sv_out)
		self._boundary.set_outermost_layers_x(u_out, raw_state['x_velocity_at_u_locations'])
		self._boundary.set_outermost_layers_y(v_out, raw_state['y_velocity_at_v_locations'])

		# Diagnose the pressure, the Exner function, the Montgomery potential
		# and the geometric height of the half levels
		pt = raw_state['air_pressure_on_interface_levels'][0, 0, 0]
		p_out, exn_out, mtg_out, h_out = self._diagnostic.get_diagnostic_variables(s_out, pt)

		# Allocate memory to store the isentropic density of all water constituents
		if not hasattr(self, '_qv_out'):
			self._qv_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qc_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qr_out = np.zeros((nx, ny, nz), dtype=dtype)

		# Diagnose the mass fraction of all water constituents
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqv_out, self._qv_out, clipping=True)
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqc_out, self._qc_out, clipping=True)
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqr_out, self._qr_out, clipping=True)

		# Diagnose the air density and temperature
		rho_out  = self._diagnostic.get_air_density(s_out, h_out)
		temp_out = self._diagnostic.get_air_temperature(exn_out)

		# Instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_density': rho_out,
			'air_isentropic_density': s_out,
			'air_pressure_on_interface_levels': p_out,
			'air_temperature': temp_out,
			'exner_function_on_interface_levels': exn_out,
			'height_on_interface_levels': h_out,
			'mass_fraction_of_water_vapor_in_air': self._qv_out,
			'mass_fraction_of_cloud_liquid_water_in_air': self._qc_out,
			'mass_fraction_of_precipitation_water_in_air': self._qr_out,
			'montgomery_potential': mtg_out,
			'x_momentum_isentropic': su_out,
			'x_velocity_at_u_locations': u_out,
			'y_momentum_isentropic': sv_out,
			'y_velocity_at_v_locations': v_out,
		}

		if self._sedimentation_on:
			if np.any(qr_now > 0.) or np.any(self._qr_out > 0.):
				# Integrate the sedimentation flux
				raw_state_out_ = self._prognostic.step_integrating_sedimentation_flux(
					stage, timestep, raw_state, raw_state_out)

				# Update the output state and the output diagnostics
				raw_state_out.update(raw_state_out_)
			else:
				raw_state_out['accumulated_precipitation'] = raw_state['accumulated_precipitation']
				raw_state_out['precipitation'] = np.zeros((nx, ny, 1))

		return raw_state_out
