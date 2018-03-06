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

		# Perform the prognostic step
		state_new = self._prognostic(dt, state, diagnostics)

		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_state_ref'):
				self._s_ref = state['isentropic_density'].values[:,:,:,0]
				self._U_ref = state['x_momentum_isentropic'].values[:,:,:,0]
				self._V_ref = state['y_momentum_isentropic'].values[:,:,:,0]

			# Extract the prognostic model variables
			s_now = state['isentropic_density'].values[:,:,:,0]
			s_new = state_new['isentropic_density'].values[:,:,:,0]
			U_now = state['x_momentum_isentropic'].values[:,:,:,0]
			U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
			V_now = state['y_momentum_isentropic'].values[:,:,:,0]
			V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:] = self._damper.apply(dt, s_now, s_new, self._s_ref)
			U_new[:,:,:] = self._damper.apply(dt, U_now, U_new, self._U_ref)
			V_new[:,:,:] = self._damper.apply(dt, V_now, V_new, self._V_ref)

			# Update the state
			upd = GridData(time_now + dt, self._grid, isentropic_density = s_new,
						   x_momentum_isentropic = U_new, y_momentum_isentropic = V_new)
			state_new.update(upd)

		if self._ismooth:
			if not self._idamp:
				# Extract the prognostic model variables
				s_now = state['isentropic_density'].values[:,:,:,0]
				s_new = state_new['isentropic_density'].values[:,:,:,0]
				U_now = state['x_momentum_isentropic'].values[:,:,:,0]
				U_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
				V_now = state['y_momentum_isentropic'].values[:,:,:,0]
				V_new = state_new['y_momentum_isentropic'].values[:,:,:,0]
			
			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			U_new[:,:,:] = self._smoother.apply(U_new)
			V_new[:,:,:] = self._smoother.apply(V_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(U_new, U_now)
			self._boundary.apply(V_new, V_now)

			# Update the state
			upd = GridData(time_now + dt, self._grid, isentropic_density = s_new,
						   x_momentum_isentropic = U_new, y_momentum_isentropic = V_new)
			state_new.update(upd)

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
		# Extract the Numpy arrays representing the current solution
		s_now   = state['isentropic_density'].values[:,:,:,0]
		u_now   = state['x_velocity'].values[:,:,:,0]
		U_now   = state['x_momentum_isentropic'].values[:,:,:,0]
		v_now   = state['y_velocity'].values[:,:,:,0]
		V_now   = state['y_momentum_isentropic'].values[:,:,:,0]
		p_now   = state['pressure'].values[:,:,:,0]
		mtg_now = state['montgomery_potential'].values[:,:,:,0]
		qv_now  = state['water_vapor'].values[:,:,:,0]
		qc_now  = state['cloud_water'].values[:,:,:,0]
		qr_now  = state['precipitation_water'].values[:,:,:,0]

		# Diagnosis the isentropic density of each water constituent
		Qv_now, Qc_now, Qr_now = self._diagnostic.get_water_constituents_isentropic_density(s_now, qv_now, qc_now, qr_now)

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_now_   = self._boundary.from_physical_to_computational_domain(s_now)
		u_now_   = self._boundary.from_physical_to_computational_domain(u_now)
		v_now_   = self._boundary.from_physical_to_computational_domain(v_now)
		mtg_now_ = self._boundary.from_physical_to_computational_domain(mtg_now)
		U_now_   = self._boundary.from_physical_to_computational_domain(U_now)
		V_now_   = self._boundary.from_physical_to_computational_domain(V_now)
		Qv_now_  = self._boundary.from_physical_to_computational_domain(Qv_now)
		Qc_now_  = self._boundary.from_physical_to_computational_domain(Qc_now)
		Qr_now_  = self._boundary.from_physical_to_computational_domain(Qr_now)

		# If the time integrator is a two time-levels method and this is the first time step:
		# assume the old solution coincides with the current one
		if not hasattr(self, '_s_old_'):
			self._s_old_  = s_now_ if self._prognostic.time_levels == 2 else None
			self._U_old_  = U_now_ if self._prognostic.time_levels == 2 else None
			self._V_old_  = V_now_ if self._prognostic.time_levels == 2 else None
			self._Qv_old_ = Qv_now_ if self._prognostic.time_levels == 2 else None
			self._Qc_old_ = Qr_now_ if self._prognostic.time_levels == 2 else None
			self._Qr_old_ = Qc_now_ if self._prognostic.time_levels == 2 else None

		# Perform the prognostic step
		s_new_, U_new_, V_new_, Qv_new_, Qc_new_, Qr_new_ = \
			self._prognostic.step_forward(dt, s_now_, u_now_, v_now_, p_now, mtg_now_, U_now_, V_now_, Qv_now_, Qc_now_, Qr_now_,
										  old_s = self._s_old_, old_U = self._U_old_, old_V = self._V_old_,
										  old_Qv = self._Qv_old_, old_Qc = self._Qc_old_, old_Qr = self._Qr_old_)

		# Bring the vectors back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new = self._boundary.from_computational_to_physical_domain(s_new_, (nx, ny, nz))
		if type(self._boundary) == RelaxedSymmetricXZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = False)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = True) 
		elif type(self._boundary) == RelaxedSymmetricYZ:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz), change_sign = True)
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self._boundary.from_computational_to_physical_domain(U_new_, (nx, ny, nz))
			V_new = self._boundary.from_computational_to_physical_domain(V_new_, (nx, ny, nz)) 
		Qv_new = self._boundary.from_computational_to_physical_domain(Qv_new_, (nx, ny, nz))
		Qc_new = self._boundary.from_computational_to_physical_domain(Qc_new_, (nx, ny, nz))
		Qr_new = self._boundary.from_computational_to_physical_domain(Qr_new_, (nx, ny, nz))
		
		# Apply the lateral boundary conditions to the conservative variables
		self._boundary.apply(s_new , s_now )
		self._boundary.apply(U_new , U_now )
		self._boundary.apply(V_new , V_now )
		self._boundary.apply(Qv_new, Qv_now)
		self._boundary.apply(Qc_new, Qc_now)
		self._boundary.apply(Qr_new, Qr_now)

		# Apply vertical damping
		if self._idamp:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref  = s_now
				self._U_ref  = U_now
				self._V_ref  = V_now
				self._Qv_ref = Qv_now
				self._Qc_ref = Qc_now
				self._Qr_ref = Qr_now

			s_new[:,:,:]  = self._damper.apply(dt, s_now , s_new , self._s_ref )
			U_new[:,:,:]  = self._damper.apply(dt, U_now , U_new , self._U_ref )
			V_new[:,:,:]  = self._damper.apply(dt, V_now , V_new , self._V_ref )
			Qv_new[:,:,:] = self._damper.apply(dt, Qv_now, Qv_new, self._Qv_ref)
			Qc_new[:,:,:] = self._damper.apply(dt, Qc_now, Qc_new, self._Qc_ref)
			Qr_new[:,:,:] = self._damper.apply(dt, Qr_now, Qr_new, self._Qr_ref)

		# Apply numerical smoothing
		if self._ismooth:
			s_new[:,:,:] = self._smoother.apply(s_new)
			U_new[:,:,:] = self._smoother.apply(U_new)
			V_new[:,:,:] = self._smoother.apply(V_new)

			self._boundary.apply(s_new , s_now )
			self._boundary.apply(U_new , U_now )
			self._boundary.apply(V_new , V_now )

		if self._ismooth_moist:
			Qv_new[:,:,:] = self._smoother_moist.apply(Qv_new)
			Qc_new[:,:,:] = self._smoother_moist.apply(Qc_new)
			Qr_new[:,:,:] = self._smoother_moist.apply(Qr_new)

			self._boundary.apply(Qv_new, Qv_now)
			self._boundary.apply(Qc_new, Qc_now)
			self._boundary.apply(Qr_new, Qr_now)

		# Diagnose the non-conservative model variables
		u_new, v_new = self._diagnostic.get_velocity_components(s_new, U_new, V_new)
		qv_new, qc_new, qr_new = self._diagnostic.get_water_constituents_mass_fraction(s_new, Qv_new, Qc_new, Qr_new)

		# Apply the lateral boundary conditions to the velocity components
		self._boundary.set_outermost_layers_x(u_new, u_now) 
		self._boundary.set_outermost_layers_y(v_new, v_now) 

		# Diagnose the pressure, the Exner function, the Montgomery potential, and the geometric height at the half levels
		p_new, exn_new, mtg_new, h_new = self._diagnostic.get_diagnostic_variables(s_new, p_now[0,0,0])

		# Update the old time step
		if self._prognostic.time_levels == 2:
			self._s_old_[:,:,:]  = s_now_
			self._U_old_[:,:,:]  = U_now_
			self._V_old_[:,:,:]  = V_now_
			self._Qv_old_[:,:,:] = Qv_now_
			self._Qc_old_[:,:,:] = Qc_now_
			self._Qr_old_[:,:,:] = Qr_now_

		# Build up the new state, and return
		state_new = StateIsentropic(state.time + dt, self._grid,
									s_new, u_new, U_new, v_new, V_new, p_new, exn_new, mtg_new, h_new,
									qv_new, qc_new, qr_new)

		return state_new
		
