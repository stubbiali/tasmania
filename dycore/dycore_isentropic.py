# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import math
import numpy as np

from dycore.diagnostic_isentropic import DiagnosticIsentropic
from dycore.dycore import DynamicalCore
from dycore.horizontal_boundary import HorizontalBoundary, RelaxedSymmetricXZ, RelaxedSymmetricYZ
from dycore.horizontal_smoothing import HorizontalSmoothing
from dycore.prognostic_isentropic import PrognosticIsentropic
from dycore.vertical_damping import VerticalDamping
import gridtools as gt
from namelist import cp, datatype, g, p_ref, Rd
from storages.grid_data import GridData
from storages.state_isentropic import StateIsentropic
from storages.state_isentropic_conservative import StateIsentropicConservative
import utils.utils_meteo as utils_meteo

class DynamicalCoreIsentropic(DynamicalCore):
	"""
	This class inherits :class:`~dycore.dycore.DynamicalCore` to implement the three-dimensional 
	(moist) isentropic dynamical core using GT4Py's stencils. The class offers different numerical
	schemes to carry out the prognostic step of the dynamical core, and supports different types of 
	lateral boundary conditions.
	"""
	def __init__(self, time_scheme, flux_scheme, horizontal_boundary_type, grid, imoist, backend,
				 idamp = True, damp_type = 'rayleigh', damp_depth = 15, damp_max = 0.0002, 
				 ismooth = True, smooth_type = 'first_order', smooth_damp_depth = 10, 
				 smooth_coeff = .03, smooth_coeff_max = .24, 
				 ismooth_moist = False, smooth_moist_type = 'first_order', smooth_moist_damp_depth = 10,
				 smooth_coeff_moist = .03, smooth_coeff_moist_max = .24):
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
		imoist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils implementing the dynamical core.
		idamp : `bool`, optional
			:obj:`True` if vertical damping is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. Default is 'rayleigh'.
			See :class:`dycore.vertical_damping.VerticalDamping` for the available options.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Default is 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Default is 0.0002.
		ismooth : `bool`, optional
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
		ismooth_moist : `bool`, optional
			:obj:`True` if numerical smoothing on water constituents is enabled, :obj:`False` otherwise. Default is :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply to the water constituents. Default is 'first-order'.
			See :class:`dycore.horizontal_smoothing.HorizontalSmoothing` for the available options.
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the water constituents. Default is 10.
		smooth_coeff_moist : `float`, optional
			Smoothing coefficient for the water constituents. Default is 0.03.
		smooth_coeff_moist_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents. Default is 0.24. 
			See :class:`~dycore.horizontal_smoothing.HorizontalSmoothing` for further details.
		"""
		# Keep track of the input parameters
		self._grid, self._imoist, self._idamp, self._ismooth, self._ismooth_moist = grid, imoist, idamp, ismooth, ismooth_moist

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = PrognosticIsentropic.factory(time_scheme, flux_scheme, grid, imoist, backend)
		nb = self._prognostic.nb

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid, nb)
		self._prognostic.boundary = self._boundary

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = DiagnosticIsentropic(grid, imoist, backend)
		self._diagnostic.boundary = self._boundary
		self._prognostic.diagnostic = self._diagnostic

		# Instantiate the class in charge of applying vertical damping
		if idamp: 
			self._damper = VerticalDamping.factory(damp_type, grid, damp_depth, damp_max, backend)

		# Instantiate the classes in charge of applying numerical smoothing
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if ismooth:
			self._smoother = HorizontalSmoothing.factory(smooth_type, (nx, ny, nz), grid, smooth_damp_depth, 
														 smooth_coeff, smooth_coeff_max, backend)
			if imoist and ismooth_moist:
				self._smoother_moist = HorizontalSmoothing.factory(smooth_moist_type, (nx, ny, nz), grid, smooth_moist_damp_depth, 
														 		   smooth_coeff_moist, smooth_coeff_moist_max, backend)

		# Set the pointer to the entry-point method, distinguishing between dry and moist model
		self._integrate = self._integrate_moist if imoist else self._integrate_dry

	def __call__(self, dt, state, diagnostics = None):
		"""
		Call operator advancing the state variables one step forward. 

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing the possibly required diagnostics. Default is :obj:`None`.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		return self._integrate(dt, state, diagnostics)

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
			if self._imoist:
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
			if self._imoist:
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
			if self._imoist:
				qv = np.zeros((nx, ny, nz), dtype = datatype)
				qc = np.zeros((nx, ny, nz), dtype = datatype)
				qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Assemble the initial state
		if self._imoist:
			state = StateIsentropic(initial_time, self._grid, s, u, U, v, V, p, exn, mtg, h, qv, qc, qr)
		else:
			state = StateIsentropic(initial_time, self._grid, s, u, U, v, V, p, exn, mtg, h)

		return state

	def _integrate_dry(self, dt, state, diagnostics):
		"""
		Method advancing the dry isentropic state by a single time step.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing the possibly required diagnostics. Default is :obj:`None`.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		# Extract the current time
		time_now = state.get_time()

		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if self._idamp or self._ismooth:
			s_now = np.copy(state['isentropic_density'].values[:,:,:,0])
			U_now = np.copy(state['x_momentum_isentropic'].values[:,:,:,0])
			V_now = np.copy(state['y_momentum_isentropic'].values[:,:,:,0])

		# Perform the prognostic step
		state_new = self._prognostic(dt, state, diagnostics)

		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref = s_now
				self._U_ref = U_now
				self._V_ref = V_now

			# Extract the prognostic model variables
			s_new = state_new['isentropic_density'].values[:,:,:,0]
			U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
			V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:] = self._damper.apply(dt, s_now, s_new, self._s_ref)
			U_new[:,:,:] = self._damper.apply(dt, U_now, U_new, self._U_ref)
			V_new[:,:,:] = self._damper.apply(dt, V_now, V_new, self._V_ref)

		if self._ismooth:
			if not self._idamp:
				# Extract the prognostic model variables
				s_new = state_new['isentropic_density'].values[:,:,:,0]
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
		state_new.update(self._diagnostic.get_velocity_components(state_new))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		state_new.update(self._diagnostic.get_diagnostic_variables(state_new))

		return state_new

	def _integrate_moist(self, dt, state, diagnostics = None):
		"""
		Method advancing the moist isentropic state by a single time step.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
		diagnostics : `obj`, optional 
			:class:`~storages.grid_data.GridData` storing the possibly required diagnostics. Default is :obj:`None`.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` representing the state at the next time level.
		"""
		# Extract the current time
		time_now = state.get_time()

		# Diagnose the isentropic density for each water constituent to build the conservative state
		wcs_is = self._diagnostic.get_water_constituents_isentropic_density(state)
		state_cons = StateIsentropicConservative(time_now, self._grid, 
			isentropic_density = state['isentropic_density'].values[:,:,:,0], 
			x_velocity = state['x_velocity'].values[:,:,:,0], 
			x_momentum_isentropic = state['x_momentum_isentropic'].values[:,:,:,0], 
			y_velocity = state['y_velocity'].values[:,:,:,0], 
			y_momentum_isentropic = state['y_momentum_isentropic'].values[:,:,:,0], 
			pressure = state['pressure'].values[:,:,:,0], 
			exner_function = state['exner_function'].values[:,:,:,0], 
			montgomery_potential = state['montgomery_potential'].values[:,:,:,0], 
			height = state['height'].values[:,:,:,0],
			water_vapor_isentropic_density = wcs_is['water_vapor_isentropic_density'].values[:,:,:,0], 
			cloud_water_isentropic_density = wcs_is['cloud_water_isentropic_density'].values[:,:,:,0], 
			precipitation_water_isentropic_density = wcs_is['precipitation_water_isentropic_density'].values[:,:,:,0])

		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if self._idamp or self._ismooth:
			s_now  = np.copy(state['isentropic_density'].values[:,:,:,0])
			U_now  = np.copy(state['x_momentum_isentropic'].values[:,:,:,0])
			V_now  = np.copy(state['y_momentum_isentropic'].values[:,:,:,0])
			Qv_now = np.copy(wcs_is['water_vapor_isentropic_density'].values[:,:,:,0])
			Qc_now = np.copy(wcs_is['cloud_water_isentropic_density'].values[:,:,:,0])
			Qr_now = np.copy(wcs_is['precipitation_water_isentropic_density'].values[:,:,:,0])

		# Perform the prognostic step
		state_cons_new = self._prognostic(dt, state_cons, diagnostics)

		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref  = s_now
				self._U_ref  = U_now
				self._V_ref  = V_now
				self._Qv_ref = Qv_now
				self._Qc_ref = Qc_now
				self._Qr_ref = Qr_now

			# Extract the prognostic model variables
			s_new  = state_cons_new['isentropic_density'].values[:,:,:,0]
			U_new  = state_cons_new['x_momentum_isentropic'].values[:,:,:,0]
			V_new  = state_cons_new['y_momentum_isentropic'].values[:,:,:,0]
			Qv_new = state_cons_new['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_new = state_cons_new['cloud_water_isentropic_density'].values[:,:,:,0]
			Qr_new = state_cons_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:]  = self._damper.apply(dt, s_now, s_new, self._s_ref)
			U_new[:,:,:]  = self._damper.apply(dt, U_now, U_new, self._U_ref)
			V_new[:,:,:]  = self._damper.apply(dt, V_now, V_new, self._V_ref)
			Qv_new[:,:,:] = self._damper.apply(dt, Qv_now, Qv_new, self._Qv_ref)
			Qc_new[:,:,:] = self._damper.apply(dt, Qc_now, Qc_new, self._Qc_ref)
			Qr_new[:,:,:] = self._damper.apply(dt, Qr_now, Qr_new, self._Qr_ref)

		if self._ismooth:
			if not self._idamp:
				# Extract the dry prognostic model variables
				s_new = state_cons_new['isentropic_density'].values[:,:,:,0]
				U_new = state_cons_new['x_momentum_isentropic'].values[:,:,:,0]
				V_new = state_cons_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			U_new[:,:,:] = self._smoother.apply(U_new)
			V_new[:,:,:] = self._smoother.apply(V_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(U_new, U_now)
			self._boundary.apply(V_new, V_now)

		if self._ismooth_moist:
			if not self._idamp:
				# Extract the moist prognostic model variables
				Qv_new = state_cons_new['water_vapor_isentropic_density'].values[:,:,:,0]
				Qc_new = state_cons_new['cloud_water_isentropic_density'].values[:,:,:,0]
				Qr_new = state_cons_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply horizontal smoothing
			Qv_new[:,:,:] = self._smoother_moist.apply(Qv_new)
			Qc_new[:,:,:] = self._smoother_moist.apply(Qc_new)
			Qr_new[:,:,:] = self._smoother_moist.apply(Qr_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(Qv_new, Qv_now)
			self._boundary.apply(Qc_new, Qc_now)
			self._boundary.apply(Qr_new, Qr_now)

		# Diagnose the mass fraction of each water constituent to build the non-conservative state
		wcs_mf = self._diagnostic.get_water_constituents_mass_fraction(state_cons_new) 
		state_new = StateIsentropic(time_now + dt, self._grid, 
			isentropic_density = state_cons_new['isentropic_density'].values[:,:,:,0], 
			x_velocity = state_cons_new['x_velocity'].values[:,:,:,0], 
			x_momentum_isentropic = state_cons_new['x_momentum_isentropic'].values[:,:,:,0], 
			y_velocity = state_cons_new['y_velocity'].values[:,:,:,0], 
			y_momentum_isentropic = state_cons_new['y_momentum_isentropic'].values[:,:,:,0], 
			pressure = state_cons_new['pressure'].values[:,:,:,0], 
			exner_function = state_cons_new['exner_function'].values[:,:,:,0], 
			montgomery_potential = state_cons_new['montgomery_potential'].values[:,:,:,0], 
			height = state_cons_new['height'].values[:,:,:,0],
			water_vapor_mass_fraction = wcs_mf['water_vapor_mass_fraction'].values[:,:,:,0], 
			cloud_water_mass_fraction = wcs_mf['cloud_water_mass_fraction'].values[:,:,:,0], 
			precipitation_water_mass_fraction = wcs_mf['precipitation_water_mass_fraction'].values[:,:,:,0])

		# Diagnose the velocity components
		state_new.update(self._diagnostic.get_velocity_components(state_new))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		state_new.update(self._diagnostic.get_diagnostic_variables(state_new))

		# Diagnose the density
		#state_new.update(self._diagnostic.get_density(state_new))

		# Diagnose the temperature
		#state_new.update(self._diagnostic.get_temperature(state_new))

		return state_new
