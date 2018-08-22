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
        :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the initial state.
        It contains the following variables:

        * air_density (unstaggered);
        * air_isentropic_density (unstaggered);
        * x_velocity (:math:`x`-staggered);
        * x_momentum_isentropic (unstaggered);
        * y_velocity (:math:`y`-staggered);
        * y_momentum_isentropic (unstaggered);
        * air_pressure_on_interface_levels (:math:`z`-staggered);
        * exner_function_on_interface_levels (:math:`z`-staggered);
        * montgomery_potential (unstaggered);
        * height_on_interface_levels (:math:`z`-staggered);
        * air_temperature (unstaggered);
        * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
        * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
        * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).
    """
	# Shortcuts
	nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
	z, z_hl, dz = self._grid.z.values, self._grid.z_on_interface_levels.values, self._grid.dz
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
			theta_hl = np.tile(self._grid.z_on_interface_levels.values[np.newaxis, np.newaxis, :], (nx, ny, 1))
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
							air_isentropic_density             = s,
							x_velocity                         = u,
							x_momentum_isentropic              = U,
							y_velocity                         = v,
							y_momentum_isentropic              = V,
							air_pressure_on_interface_levels   = p,
							exner_function_on_interface_levels = exn,
							montgomery_potential               = mtg,
							height_on_interface_levels         = h)

	# Diagnose the air density and temperature
	state.extend(self._diagnostic.get_air_density(state)),
	state.extend(self._diagnostic.get_air_temperature(state))

	if self._moist_on:
		# Add the mass fraction of each water component
		state.add_variables(initial_time,
							mass_fraction_of_water_vapor_in_air = qv,
							mass_fraction_of_cloud_liquid_water_in_air = qc,
							mass_fraction_of_precipitation_water_in_air = qr)

	return state

