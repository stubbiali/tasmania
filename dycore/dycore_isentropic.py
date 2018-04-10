import math
import numpy as np

from tasmania.dycore.diagnostic_isentropic import DiagnosticIsentropic
from tasmania.dycore.dycore import DynamicalCore
from tasmania.dycore.horizontal_boundary import HorizontalBoundary
from tasmania.dycore.horizontal_smoothing import HorizontalSmoothing
from tasmania.dycore.prognostic_isentropic import PrognosticIsentropic
from tasmania.dycore.vertical_damping import VerticalDamping
import gridtools as gt
from tasmania.namelist import cp, datatype, g, p_ref, Rd
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic
import tasmania.utils.utils as utils
import tasmania.utils.utils_meteo as utils_meteo

class DynamicalCoreIsentropic(DynamicalCore):
	"""
	This class inherits :class:`~dycore.dycore.DynamicalCore` to implement the three-dimensional 
	(moist) isentropic dynamical core using GT4Py stencils. The class offers different numerical
	schemes to carry out the prognostic step of the dynamical core, and supports different types of 
	lateral boundary conditions.
	"""
	def __init__(self, time_scheme, flux_scheme, horizontal_boundary_type, grid, moist_on, backend,
				 damp_on = True, damp_type = 'rayleigh', damp_depth = 15, damp_max = 0.0002, 
				 smooth_on = True, smooth_type = 'first_order', smooth_damp_depth = 10, 
				 smooth_coeff = .03, smooth_coeff_max = .24, 
				 smooth_moist_on = False, smooth_moist_type = 'first_order', smooth_moist_damp_depth = 10,
				 smooth_moist_coeff = .03, smooth_moist_coeff_max = .24,
				 physics_dynamics_coupling_on = False, sedimentation_on = False):
		"""
		Constructor.

		Parameters
		----------
		time_scheme : str
			String specifying the time stepping method to implement.		
			See :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` for the available options.	
		flux_scheme : str 
			String specifying the numerical fluc to use.
			See :class:`~dycore.prognostic_flux.PrognosticFlux` for the available options.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions. 
			See :class:`~dycore.horizontal_boundary.HorizontalBoundary` for the available options.
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils implementing the dynamical core.
		damp_on : `bool`, optional
			:obj:`True` if vertical damping is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. Default is 'rayleigh'.
			See :class:`dycore.vertical_damping.VerticalDamping` for the available options.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Default is 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Default is 0.0002.
		smooth_on : `bool`, optional
			:obj:`True` if numerical smoothing is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement. Default is 'first-order'.
			See :class:`dycore.horizontal_smoothing.HorizontalSmoothing` for the available options.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Default is 10.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Default is 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. Default is 0.24. 
			See :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` for further details.
		smooth_moist_on : `bool`, optional
			:obj:`True` if numerical smoothing on water constituents is enabled, :obj:`False` otherwise. 
			Default is :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply to the water constituents. Default is 'first-order'.
			See :class:`dycore.horizontal_smoothing.HorizontalSmoothing` for the available options.
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the water constituents. Default is 10.
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Default is 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents. Default is 0.24. 
			See :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` for further details.
		physics_dynamics_coupling_on : `bool`, optional
			:obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential 
			temperature, :obj:`False` otherwise. Default is :obj:`False`.
		sedimentation_on : `bool`, optional
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise. Default is :obj:`False`.
		"""
		# Call parent constructor
		super().__init__(grid, moist_on)

		# Keep track of the input parameters
		self._damp_on, self._smooth_on, self._smooth_moist_on = damp_on, smooth_on, smooth_moist_on
		self._physics_dynamics_coupling_on = physics_dynamics_coupling_on
		self._sedimentation_on = sedimentation_on

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = PrognosticIsentropic.factory(time_scheme, flux_scheme, grid, moist_on, backend,
														physics_dynamics_coupling_on, sedimentation_on)
		nb = self._prognostic.nb

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid, nb)
		self._prognostic.boundary = self._boundary

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = DiagnosticIsentropic(grid, moist_on, backend)
		self._diagnostic.boundary = self._boundary
		self._prognostic.diagnostic = self._diagnostic

		# Instantiate the class in charge of applying vertical damping
		if damp_on: 
			self._damper = VerticalDamping.factory(damp_type, grid, damp_depth, damp_max, backend)

		# Instantiate the classes in charge of applying numerical smoothing
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if smooth_on:
			self._smoother = HorizontalSmoothing.factory(smooth_type, (nx, ny, nz), grid, smooth_damp_depth, 
														 smooth_coeff, smooth_coeff_max, backend)
			if moist_on and smooth_moist_on:
				self._smoother_moist = HorizontalSmoothing.factory(smooth_moist_type, (nx, ny, nz), grid, 
																   smooth_moist_damp_depth, 
														 		   smooth_moist_coeff, smooth_moist_coeff_max, backend)

		# Set the pointer to the entry-point method, distinguishing between dry and moist model
		self._step = self._step_dry if not moist_on else self._step_moist

	@property
	def time_levels(self):
		"""
		Get the number of time leves the dynamical core relies on.

		Return
		------
		int :
			The number of time levels needed by the dynamical core.
		"""
		return self._prognostic.time_levels

	@property
	def microphysics(self):
		"""
		Get the attribute in charge of computing the raindrop fall velocity.

		Return
		------
		obj :
			Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
			or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` which provides the raindrop fall velocity.
		"""
		return self._microphysics

	@microphysics.setter
	def microphysics(self, micro):
		"""
		Set the attribute in charge of computing the raindrop fall velocity.

		Parameters
		----------
		micro : obj
			Instance of a derived class of either 
			:class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics` or 
			:class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` which provides the 
			raindrop fall velocity.
		"""
		# Set attribute
		self._microphysics = micro

		# Update prognostic attribute
		self._prognostic.microphysics = micro

	def __call__(self, dt, state, diagnostics = None, tendencies = None):
		"""
		Call operator advancing the state variables one step forward. 

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing diagnostics, namely:
			
			* change_over_time_in_air_potential_temperature (unstaggered, required when coupling between \
				physics and dynamics is switched on);
			* accumulated_precipitation (unstaggered).

			Default is :obj:`None`.
		tendencies : `obj`, optional
			:class:`~storages.grid_data.GridData` storing tendencies. Default is obj:`None`.

		Return
		------
		state_new : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
			It contains the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height (:math:`z`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered);
			* air_density (unstaggered, only if cloud microphysics is switched on);
			* air_temperature (unstaggered, only if cloud microphysics is switched on).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` storing output diagnostics, namely:

			* precipitation (unstaggered, only if rain sedimentation is switched on);
			* accumulated_precipitation (unstaggered, only if rain sedimentation is switched on);
		"""
		return self._step(dt, state, diagnostics, tendencies)

	def get_initial_state(self, initial_time, initial_state_type, **kwargs):
		"""
		Get the initial state, based on the identifier :data:`initial_state_type`. Particularly:

		* if :data:`initial_state_type == 0`:

			- :math:`u(x, \, y, \, \\theta, \, 0) = u_0` and :math:`v(x, \, y, \, \\theta, \, 0) = v_0`;
			- the Exner function, the pressure, the Montgomery potential, the height of the isentropes, \
				and the isentropic density are derived from the Brunt-Vaisala frequency :math:`N`;
			- the mass fraction of water vapor is derived from the relative humidity, which is horizontally uniform \
				and different from zero only in a band close to the surface;
			- the mass fraction of cloud water and precipitation water is zero;

		* if :data:`initial_state_type == 1`:

			- :math:`u(x, \, y, \, \\theta, \, 0) = u_0` and :math:`v(x, \, y, \, \\theta, \, 0) = v_0`;
			- the Exner function, the pressure, the Montgomery potential, the height of the isentropes, \
				and the isentropic density are derived from the Brunt-Vaisala frequency :math:`N`;
			- the mass fraction of water vapor is derived from the relative humidity, which is sinusoidal in the \
				:math:`x`-direction and uniform in the :math:`y`-direction, and different from zero only in a band \
				close to the surface;
			- the mass fraction of cloud water and precipitation water is zero.

		* if :data:`initial_state_type == 2`:

			- :math:`u(x, \, y, \, \\theta, \, 0) = u_0` and :math:`v(x, \, y, \, \\theta, \, 0) = v_0`;
			- :math:`T(x, \, y, \, \\theta, \, 0) = T_0`.

		Parameters
		----------
		initial_time : obj 
			:class:`datetime.datetime` representing the initial simulation time.
		case : int
			Identifier.

		Keyword arguments
		-----------------
		x_velocity_initial : float 
			The initial, uniform :math:`x`-velocity :math:`u_0`. Default is :math:`10 m s^{-1}`.
		y_velocity_initial : float 
			The initial, uniform :math:`y`-velocity :math:`v_0`. Default is :math:`0 m s^{-1}`.
		brunt_vaisala_initial : float
			If :data:`initial_state_type == 0`, the uniform Brunt-Vaisala frequence :math:`N`. Default is :math:`0.01`.
		temperature_initial : float
			If :data:`initial_state_type == 1`, the uniform initial temperature :math:`T_0`. Default is :math:`250 K`.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` representing the initial state.
			It contains the following variables:

			* air_density (unstaggered);
			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height (:math:`z`-staggered);
			* air_temperature (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		z, z_hl, dz = self._grid.z.values, self._grid.z_half_levels.values, self._grid.dz
		topo = self._grid.topography_height

		#
		# Case 0
		#
		if initial_state_type == 0:
			# The initial velocity
			u = kwargs.get('x_velocity_initial', 10.) * np.ones((nx + 1, ny, nz), dtype = datatype)
			v = kwargs.get('y_velocity_initial', 0.) * np.ones((nx, ny + 1, nz), dtype = datatype)

			# The initial Exner function
			brunt_vaisala_initial = kwargs.get('brunt_vaisala_initial', .01)
			exn_col = np.zeros(nz + 1, dtype = datatype)
			exn_col[-1] = cp
			for k in range(0, nz):
				exn_col[nz - k - 1] = exn_col[nz - k] - 4.* dz * g**2 / \
									  (brunt_vaisala_initial**2 * ((z_hl[nz - k - 1] + z_hl[nz - k])**2))
			exn = np.tile(exn_col[np.newaxis, np.newaxis, :], (nx, ny, 1))

			# The initial pressure
			p = p_ref * (exn / cp) ** (cp / Rd)

			# The initial Montgomery potential
			mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
			mtg = np.zeros((nx, ny, nz), dtype = datatype)
			mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
			for k in range(1, nz):
				mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

			# The initial geometric height of the isentropes
			h = np.zeros((nx, ny, nz + 1), dtype = datatype)
			h[:, :, -1] = self._grid.topography_height
			for k in range(0, nz):
				h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (brunt_vaisala_initial**2 * z[nz - k - 1])

			# The initial isentropic density
			s = - 1. / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

			# The initial momentums
			U = s * kwargs.get('x_velocity_initial', 10.)
			V = s * kwargs.get('y_velocity_initial', 0.)

			# The initial water constituents
			if self._moist_on:
				# Set the initial relative humidity
				rhmax   = 0.98
				kw      = 10
				kc      = 11
				k       = np.arange(kc - kw + 1, kc + kw)
				rh1d    = np.zeros(nz, dtype = datatype)
				rh1d[k] = rhmax * np.cos(np.abs(k - kc) * math.pi / (2. * kw)) ** 2
				rh1d    = rh1d[::-1]
				rh      = np.tile(rh1d[np.newaxis, np.newaxis, :], (nx, ny, 1))

				# Compute the pressure and the temperature at the main levels
				p_ml  = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
				theta = np.tile(self._grid.z.values[np.newaxis, np.newaxis, :], (nx, ny, 1))
				#T_ml  = theta * (p_ml / p_ref) ** (Rd / cp)
				theta_hl = np.tile(self._grid.z_half_levels.values[np.newaxis, np.newaxis, :], (nx, ny, 1))
				T_ml = 0.5 * (theta_hl[:, :, :-1] * exn[:, :, :-1] + theta_hl[:, :, 1:] * exn[:, :, 1:]) / cp

				# Convert relative humidity to water vapor
				qv = utils_meteo.convert_relative_humidity_to_water_vapor('goff_gratch', p_ml, T_ml, rh)
				
				# Set the initial cloud water and precipitation water to zero
				qc = np.zeros((nx, ny, nz), dtype = datatype)
				qr = np.zeros((nx, ny, nz), dtype = datatype)

		#
		# Case 1
		#
		if initial_state_type == 1:
			# The initial velocity
			u = kwargs.get('x_velocity_initial', 10.) * np.ones((nx + 1, ny, nz), dtype = datatype)
			v = kwargs.get('y_velocity_initial', 0.) * np.ones((nx, ny + 1, nz), dtype = datatype)

			# The initial Exner function
			brunt_vaisala_initial = kwargs.get('brunt_vaisala_initial', .01)
			exn_col = np.zeros(nz + 1, dtype = datatype)
			exn_col[-1] = cp
			for k in range(0, nz):
				exn_col[nz - k - 1] = exn_col[nz - k] - dz * g**2 / \
									  (brunt_vaisala_initial**2 * z[nz - k - 1]**2)
			exn = np.tile(exn_col[np.newaxis, np.newaxis, :], (nx, ny, 1))

			# The initial pressure
			p = p_ref * (exn / cp) ** (cp / Rd)

			# The initial Montgomery potential
			mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
			mtg = np.zeros((nx, ny, nz), dtype = datatype)
			mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
			for k in range(1, nz):
				mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

			# The initial geometric height of the isentropes
			h = np.zeros((nx, ny, nz + 1), dtype = datatype)
			h[:, :, -1] = self._grid.topography_height
			for k in range(0, nz):
				h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (brunt_vaisala_initial**2 * z[nz - k - 1])

			# The initial isentropic density
			s = - 1. / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

			# The initial momentums
			U = s * kwargs.get('x_velocity_initial', 10.)
			V = s * kwargs.get('y_velocity_initial', 0.)

			# The initial water constituents
			if self._moist_on:
				# Set the initial relative humidity
				rhmax   = 0.98
				kw      = 10
				kc      = 11
				k       = np.arange(kc - kw + 1, kc + kw)
				rh1d    = np.zeros(nz, dtype = datatype)
				rh1d[k] = rhmax * np.cos(np.abs(k - kc) * math.pi / (2. * kw)) ** 2
				rh1d    = rh1d[::-1]
				rh      = np.tile(rh1d[np.newaxis, np.newaxis, :], (nx, ny, 1))

				# Compute the pressure and the temperature at the main levels
				p_ml  = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
				theta = np.tile(self._grid.z.values[np.newaxis, np.newaxis, :], (nx, ny, 1))
				T_ml  = theta * (p_ml / p_ref) ** (Rd / cp)

				# Convert relative humidity to water vapor
				qv = utils_meteo.convert_relative_humidity_to_water_vapor('goff_gratch', p_ml, T_ml, rh)

				# Make the distribution of water vapor x-periodic
				x		= np.tile(self._grid.x.values[:, np.newaxis, np.newaxis], (1, ny, nz))
				xl, xr  = x[0, 0, 0], x[-1, 0, 0]
				qv      = qv * (2. + np.sin(2. * math.pi * (x - xl) / (xr - xl)))
				
				# Set the initial cloud water and precipitation water to zero
				qc = np.zeros((nx, ny, nz), dtype = datatype)
				qr = np.zeros((nx, ny, nz), dtype = datatype)

		#
		# Case 2
		#
		if initial_state_type == 2:
			# The initial velocity
			u = kwargs.get('x_velocity_initial', 10.) * np.ones((nx + 1, ny, nz), dtype = datatype)
			v = kwargs.get('y_velocity_initial', 0.) * np.ones((nx, ny + 1, nz), dtype = datatype)

			# The initial pressure
			p = p_ref * (kwargs.get('temperature_initial', 250) / \
				np.tile(z_hl[np.newaxis, np.newaxis, :], (nx, ny, 1))) ** (cp / Rd)

			# The initial Exner function
			exn = cp * (p / p_ref) ** (Rd / cp)

			# The initial Montgomery potential
			mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
			mtg = np.zeros((nx, ny, nz), dtype = datatype)
			mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
			for k in range(1, nz):
				mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

			# The initial geometric height of the isentropes
			h = np.zeros((nx, ny, nz + 1), dtype = datatype)
			h[:, :, -1] = self._grid.topography_height
			for k in range(0, nz):
				h[:, :, nz - k - 1] = h[:, :, nz - k] - (p[:, :, nz - k - 1] - p[:, :, nz - k]) * Rd / (cp * g) * \
									  z_hl[nz - k] * exn[:, :, nz - k] / p[:, :, nz - k]

			# The initial isentropic density
			s = - 1. / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

			# The initial momentums
			U = s * kwargs.get('x_velocity_initial', 10.)
			V = s * kwargs.get('y_velocity_initial', 0.)

			# The initial water constituents
			if self._moist_on:
				qv = np.zeros((nx, ny, nz), dtype = datatype)
				qc = np.zeros((nx, ny, nz), dtype = datatype)
				qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Assemble the initial state
		state = StateIsentropic(initial_time, self._grid, 
								air_isentropic_density = s, 
								x_velocity = u, 
								x_momentum_isentropic = U, 
								y_velocity = v, 
								y_momentum_isentropic = V, 
								air_pressure = p, 
								exner_function = exn, 
								montgomery_potential = mtg, 
								height = h)

		# Diagnose the air density and temperature
		state.update(self._diagnostic.get_air_density(state)),
		state.update(self._diagnostic.get_air_temperature(state))

		if self._moist_on:
			# Add the mass fraction of each water component
			state.add(mass_fraction_of_water_vapor_in_air = qv, 
					  mass_fraction_of_cloud_liquid_water_in_air = qc,
					  mass_fraction_of_precipitation_water_in_air = qr)

		return state

	def _step_dry(self, dt, state, diagnostics, tendencies):
		"""
		Method advancing the dry isentropic state by a single time step.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (unstaggered).

		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing diagnostics.
			For the time being, this is not actually used. 
		tendencies : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing tendencies.
			For the time being, this is not actually used. 

		Return
		------
		state_new : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
			It contains the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height (:math:`z`-staggered).

		diagnostics_out : obj
			Empty :class:`~tasmania.storages.grid_data.GridData`, as no diagnostics are computed. 	
		"""
		# Initialize the empty GridData to return
		time_now = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		diagnostics_out = GridData(time_now + dt, self._grid)

		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if self._damp_on or self._smooth_on:
			s_now = np.copy(state['air_isentropic_density'].values[:,:,:,0])
			U_now = np.copy(state['x_momentum_isentropic'].values[:,:,:,0])
			V_now = np.copy(state['y_momentum_isentropic'].values[:,:,:,0])

		# Perform the prognostic step
		state_new = self._prognostic.step_neglecting_vertical_advection(dt, state, diagnostics = diagnostics,
																		tendencies = tendencies)

		if self._damp_on:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref = s_now
				self._U_ref = U_now
				self._V_ref = V_now

			# Extract the prognostic model variables
			s_new = state_new['air_isentropic_density'].values[:,:,:,0]
			U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
			V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:] = self._damper.apply(dt, s_now, s_new, self._s_ref)
			U_new[:,:,:] = self._damper.apply(dt, U_now, U_new, self._U_ref)
			V_new[:,:,:] = self._damper.apply(dt, V_now, V_new, self._V_ref)

		if self._smooth_on:
			if not self._damp_on:
				# Extract the prognostic model variables
				s_new = state_new['air_isentropic_density'].values[:,:,:,0]
				U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
				V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			U_new[:,:,:] = self._smoother.apply(U_new)
			V_new[:,:,:] = self._smoother.apply(V_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(U_new, U_now)
			self._boundary.apply(V_new, V_now)

		# Diagnose the velocity components
		state_new.update(self._diagnostic.get_velocity_components(state_new, state))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		state_new.update(self._diagnostic.get_diagnostic_variables(state_new, state['air_pressure'].values[0,0,0,0]))

		return state_new, diagnostics_out

	def _step_moist(self, dt, state, diagnostics, tendencies):
		"""
		Method advancing the moist isentropic state by a single time step.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing diagnostics, namely:
			
			* change_over_time_in_air_potential_temperature (unstaggered, required only if coupling \
				between physics and dynamics is switched on);
			* accumulated_precipitation (unstaggered).

		tendencies : `obj`, optional 
			:class:`~storages.grid_data.GridData` possibly storing tendencies.
			For the time being, this is not actually used.

		Return
		------
		state_new : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
			It contains the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height (:math:`z`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered);
			* air_density (unstaggered, only if cloud microphysics is switched on);
			* air_temperature (unstaggered, only if cloud microphysics is switched on).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting output diagnostics, namely:

			* precipitation (unstaggered, only if rain sedimentation is switched on);
			* accumulated_precipitation (unstaggered, only if rain sedimentation is switched on).
		"""
		# Initialize the GridData to return
		time_now = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		diagnostics_out = GridData(time_now + dt, self._grid)

		# Diagnose the isentropic density for each water constituent to build the conservative state
		state.update(self._diagnostic.get_water_constituents_isentropic_density(state))

		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if self._damp_on or self._smooth_on:
			s_now  = np.copy(state['air_isentropic_density'].values[:,:,:,0])
			U_now  = np.copy(state['x_momentum_isentropic'].values[:,:,:,0])
			V_now  = np.copy(state['y_momentum_isentropic'].values[:,:,:,0])
			Qv_now = np.copy(state['water_vapor_isentropic_density'].values[:,:,:,0])
			Qc_now = np.copy(state['cloud_liquid_water_isentropic_density'].values[:,:,:,0])
			Qr_now = np.copy(state['precipitation_water_isentropic_density'].values[:,:,:,0])

		# Perform the prognostic step, neglecting the vertical advection
		state_new = self._prognostic.step_neglecting_vertical_advection(dt, state, diagnostics = diagnostics, 
																		tendencies = tendencies)

		if self._physics_dynamics_coupling_on:
			# Couple physics with dynamics
			state_new_ = self._prognostic.step_coupling_physics_with_dynamics(dt, state, state_new, diagnostics)

			# Update the output state
			state_new.update(state_new_)

		if self._damp_on:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref  = s_now
				self._U_ref  = U_now
				self._V_ref  = V_now
				self._Qv_ref = Qv_now
				self._Qc_ref = Qc_now
				self._Qr_ref = Qr_now

			# Extract the prognostic model variables
			s_new  = state_new['air_isentropic_density'].values[:,:,:,0]
			U_new  = state_new['x_momentum_isentropic'].values[:,:,:,0]
			V_new  = state_new['y_momentum_isentropic'].values[:,:,:,0]
			Qv_new = state_new['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_new = state_new['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr_new = state_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:]  = self._damper.apply(dt, s_now, s_new, self._s_ref)
			U_new[:,:,:]  = self._damper.apply(dt, U_now, U_new, self._U_ref)
			V_new[:,:,:]  = self._damper.apply(dt, V_now, V_new, self._V_ref)
			Qv_new[:,:,:] = self._damper.apply(dt, Qv_now, Qv_new, self._Qv_ref)
			Qc_new[:,:,:] = self._damper.apply(dt, Qc_now, Qc_new, self._Qc_ref)
			Qr_new[:,:,:] = self._damper.apply(dt, Qr_now, Qr_new, self._Qr_ref)

		if self._smooth_on:
			if not self._damp_on:
				# Extract the dry prognostic model variables
				s_new = state_new['air_isentropic_density'].values[:,:,:,0]
				U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
				V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			U_new[:,:,:] = self._smoother.apply(U_new)
			V_new[:,:,:] = self._smoother.apply(V_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(U_new, U_now)
			self._boundary.apply(V_new, V_now)

		if self._smooth_moist_on:
			if not self._damp_on:
				# Extract the moist prognostic model variables
				Qv_new = state_new['water_vapor_isentropic_density'].values[:,:,:,0]
				Qc_new = state_new['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
				Qr_new = state_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply horizontal smoothing
			Qv_new[:,:,:] = self._smoother_moist.apply(Qv_new)
			Qc_new[:,:,:] = self._smoother_moist.apply(Qc_new)
			Qr_new[:,:,:] = self._smoother_moist.apply(Qr_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(Qv_new, Qv_now)
			self._boundary.apply(Qc_new, Qc_now)
			self._boundary.apply(Qr_new, Qr_now)

		# Diagnose the mass fraction of each water constituent, possibly clipping negative values
		state_new.update(self._diagnostic.get_mass_fraction_of_water_constituents_in_air(state_new)) 

		# Diagnose the velocity components
		state_new.update(self._diagnostic.get_velocity_components(state_new, state))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		state_new.update(self._diagnostic.get_diagnostic_variables(state_new, state['air_pressure'].values[0,0,0,0]))

		if self.microphysics is not None:
			# Diagnose the density
			state_new.update(self._diagnostic.get_air_density(state_new))

			# Diagnose the temperature
			state_new.update(self._diagnostic.get_air_temperature(state_new))

		if self._sedimentation_on:
			qr     = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]
			qr_new = state_new['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

			if np.any(qr > 0.) or np.any(qr_new > 0.):
				# Resolve rain sedimentation
				state_new_, diagnostics_out_ = self._prognostic.step_resolving_sedimentation(dt, state, state_new, diagnostics)

				# Update the output state and the output diagnostics
				state_new.update(state_new_)
				diagnostics_out.update(diagnostics_out_)
			else:
				diagnostics_out.add(precipitation = np.zeros((self._grid.nx, self._grid.ny), dtype = datatype),
									accumulated_precipitation = np.zeros((self._grid.nx, self._grid.ny), dtype = datatype))

		return state_new, diagnostics_out
