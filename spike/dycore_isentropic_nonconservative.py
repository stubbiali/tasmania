# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from tasmania.dycore.diagnostic_isentropic import DiagnosticIsentropic
from tasmania.dycore.dycore import DynamicalCore
from tasmania.dycore.horizontal_boundary import HorizontalBoundary
from tasmania.dycore.horizontal_smoothing import HorizontalSmoothing
from tasmania.dycore.prognostic_isentropic_nonconservative import (
    PrognosticIsentropicNonconservative,
)
from tasmania.dycore.vertical_damping import VerticalDamping
from tasmania.conf import cp, datatype, g, p_ref, Rd
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic
import python.utils.utils as utils
import python.utils.utils_meteo as utils_meteo


class DynamicalCoreIsentropicNonconservative(DynamicalCore):
    """
    This class inherits :class:`~tasmania.dycore.dycore.DynamicalCore` to implement the three-dimensional
    (moist) isentropic_prognostic dynamical core relying upon GT4Py stencils. The class offers different numerical
    schemes to carry out the prognostic step of the dynamical core, and supports different types of
    lateral boundary conditions. The nonconservative form of the governing equations is used.
    """

    def __init__(
        self,
        time_scheme,
        flux_scheme,
        horizontal_boundary_type,
        grid,
        moist_on,
        backend,
        damp_on=True,
        damp_type="rayleigh",
        damp_depth=15,
        damp_max=0.0002,
        smooth_on=True,
        smooth_type="first_order",
        smooth_damp_depth=10,
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_moist_on=False,
        smooth_moist_type="first_order",
        smooth_moist_damp_depth=10,
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        physics_dynamics_coupling_on=False,
        sedimentation_on=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        time_scheme : str
                String specifying the time stepping method to implement.
                See :class:`~tasmania.dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative`
                for the available options.
        flux_scheme : str
                String specifying the numerical flux to use.
                See :class:`~tasmania.dycore.flux_isentropic_nonconservative.FluxIsentropicNonconservative`
                for the available options.
        horizontal_boundary_type : str
                String specifying the horizontal boundary conditions.
                See :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` for the available options.
        grid : obj
                :class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
        moist_on : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        backend : obj
                :class:`gridtools.mode` specifying the backend for the GT4Py stencils implementing the dynamical core.
        damp_on : `bool`, optional
                :obj:`True` if vertical damping is enabled, :obj:`False` otherwise. Default is :obj:`True`.
        damp_type : `str`, optional
                String specifying the type of vertical damping to apply. Default is 'rayleigh'.
                See :class:`~tasmania.dycore.vertical_damping.VerticalDamping` for the available options.
        damp_depth : `int`, optional
                Number of vertical layers in the damping region. Default is 15.
        damp_max : `float`, optional
                Maximum value for the damping coefficient. Default is 0.0002.
        smooth_on : `bool`, optional
                :obj:`True` if numerical smoothing is enabled, :obj:`False` otherwise. Default is :obj:`True`.
        smooth_type: `str`, optional
                String specifying the smoothing technique to implement. Default is 'first-order'.
                See :class:`~tasmania.dycore.horizontal_smoothing.HorizontalSmoothing` for the available options.
        smooth_damp_depth : `int`, optional
                Number of vertical layers in the smoothing damping region. Default is 10.
        smooth_coeff : `float`, optional
                Smoothing coefficient. Default is 0.03.
        smooth_coeff_max : `float`, optional
                Maximum value for the smoothing coefficient. Default is 0.24.
                See :class:`~tasmania.dycore.horizontal_smoothing.HorizontalSmoothing` for further details.
        smooth_moist_on : `bool`, optional
                :obj:`True` if numerical smoothing on water constituents is enabled, :obj:`False` otherwise.
                Default is :obj:`True`.
        smooth_moist_type: `str`, optional
                String specifying the smoothing technique to apply to the water constituents. Default is 'first-order'.
                See :class:`~tasmania.dycore.horizontal_smoothing.HorizontalSmoothing` for the available options.
        smooth_moist_damp_depth : `int`, optional
                Number of vertical layers in the smoothing damping region for the water constituents. Default is 10.
        smooth_moist_coeff : `float`, optional
                Smoothing coefficient for the water constituents. Default is 0.03.
        smooth_moist_coeff_max : `float`, optional
                Maximum value for the smoothing coefficient for the water constituents. Default is 0.24.
                See :class:`~tasmania.dycore.horizontal_smoothing.HorizontalSmoothing` for further details.
        physics_dynamics_coupling_on : `bool`, optional
                :obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential
                temperature, :obj:`False` otherwise. Default is :obj:`False`.
        sedimentation_on : `bool`, optional
                :obj:`True` to account for rain sedimentation, :obj:`False` otherwise. Default is :obj:`False`.
        """
        # Call parent constructor
        super().__init__(grid, moist_on)

        # Keep track of the input parameters
        self._damp_on, self._smooth_on, self._smooth_moist_on = (
            damp_on,
            smooth_on,
            smooth_moist_on,
        )
        self._physics_dynamics_coupling_on = physics_dynamics_coupling_on
        self._sedimentation_on = sedimentation_on

        # Instantiate the class implementing the prognostic part of the dycore
        self._prognostic = PrognosticIsentropicNonconservative.factory(
            time_scheme,
            flux_scheme,
            grid,
            moist_on,
            backend,
            physics_dynamics_coupling_on,
            sedimentation_on,
        )
        nb = self._prognostic.nb

        # Instantiate the class taking care of the boundary conditions
        self._boundary = HorizontalBoundary.factory(
            horizontal_boundary_type, grid, nb
        )
        self._prognostic.boundary = self._boundary

        # Instantiate the class implementing the diagnostic part of the dycore
        self._diagnostic = DiagnosticIsentropic(grid, moist_on, backend)
        self._diagnostic.boundary = self._boundary
        self._prognostic.diagnostic = self._diagnostic

        # Instantiate the classes in charge of applying vertical damping
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        if damp_on:
            self._damper_unstg = VerticalDamping.factory(
                damp_type, (nx, ny, nz), grid, damp_depth, damp_max, backend
            )
            self._damper_stg_x = VerticalDamping.factory(
                damp_type,
                (nx + 1, ny, nz),
                grid,
                damp_depth,
                damp_max,
                backend,
            )
            self._damper_stg_y = VerticalDamping.factory(
                damp_type,
                (nx, ny + 1, nz),
                grid,
                damp_depth,
                damp_max,
                backend,
            )

        # Instantiate the classes in charge of applying numerical smoothing
        if smooth_on:
            self._smoother_s = HorizontalSmoothing.factory(
                smooth_type,
                (nx, ny, nz),
                grid,
                smooth_damp_depth,
                smooth_coeff,
                smooth_coeff_max,
                backend,
            )
            self._smoother_u = HorizontalSmoothing.factory(
                smooth_type,
                (nx + 1, ny, nz),
                grid,
                smooth_damp_depth,
                smooth_coeff,
                smooth_coeff_max,
                backend,
            )
            self._smoother_v = HorizontalSmoothing.factory(
                smooth_type,
                (nx, ny + 1, nz),
                grid,
                smooth_damp_depth,
                smooth_coeff,
                smooth_coeff_max,
                backend,
            )
            if moist_on and smooth_moist_on:
                self._smoother_q = HorizontalSmoothing.factory(
                    smooth_moist_type,
                    (nx, ny, nz),
                    grid,
                    smooth_moist_damp_depth,
                    smooth_moist_coeff,
                    smooth_moist_coeff_max,
                    backend,
                )

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
        Get the attribute taking care of the cloud microphysical dynamics.

        Return
        ------
        obj :
                Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
                or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics`.
        """
        return self._microphysics

    @microphysics.setter
    def microphysics(self, micro):
        """
        Set the attribute taking care of the cloud microphysical dynamics.

        Parameters
        ----------
        micro : obj
                Instance of a derived class of either :class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics`
                or :class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics`.
        """
        # Set attribute
        self._microphysics = micro

        # Update prognostic attribute
        self._prognostic.microphysics = micro

    def __call__(self, dt, state, diagnostics=None, tendencies=None):
        """
        Call operator advancing the state variables one step forward.

        Parameters
        ----------
        dt : obj
                :class:`datetime.timedelta` representing the time step.
        state :obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
                * montgomery_potential (unstaggered);
                * mass_fraction_of_water_vapor_in_air (unstaggered, optional);
                * mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
                * mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

        diagnostics : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` storing diagnostics, namely:

                * accumulated_precipitation (unstaggered).

                Default is :obj:`None`.
        tendencies : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` storing tendencies. Default is obj:`None`.

        Return
        ------
        state_new : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the state at the next time level.
                It contains the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * air_pressure_on_interface_levels (:math:`z`-staggered);
                * exner_function_on_interface_levels (:math:`z`-staggered);
                * montgomery_potential (unstaggered);
                * height_on_interface_levels (:math:`z`-staggered);
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
				and the isentropic_prognostic density are derived from the Brunt-Vaisala frequency :math:`N`;
			- the mass fraction of water vapor is derived from the relative humidity, which is horizontally uniform \
				and different from zero only in a band close to the surface;
			- the mass fraction of cloud water and precipitation water is zero;

		* if :data:`initial_state_type == 1`:

			- :math:`u(x, \, y, \, \\theta, \, 0) = u_0` and :math:`v(x, \, y, \, \\theta, \, 0) = v_0`;
			- the Exner function, the pressure, the Montgomery potential, the height of the isentropes, \
				and the isentropic_prognostic density are derived from the Brunt-Vaisala frequency :math:`N`;
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
			* y_velocity (:math:`y`-staggered);
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
        z, z_hl, dz = (
            self._grid.z.values,
            self._grid.z_on_interface_levels.values,
            self._grid.dz,
        )
        topo = self._grid.topography_height

        #
        # Case 0
        #
        if initial_state_type == 0:
            # The initial velocity
            u = kwargs.get("x_velocity_initial", 10.0) * np.ones(
                (nx + 1, ny, nz), dtype=datatype
            )
            v = kwargs.get("y_velocity_initial", 0.0) * np.ones(
                (nx, ny + 1, nz), dtype=datatype
            )

            # The initial Exner function
            brunt_vaisala_initial = kwargs.get("brunt_vaisala_initial", 0.01)
            exn_col = np.zeros(nz + 1, dtype=datatype)
            exn_col[-1] = cp
            for k in range(0, nz):
                exn_col[nz - k - 1] = exn_col[nz - k] - 4.0 * dz * g ** 2 / (
                    brunt_vaisala_initial ** 2
                    * ((z_hl[nz - k - 1] + z_hl[nz - k]) ** 2)
                )
            exn = np.tile(exn_col[np.newaxis, np.newaxis, :], (nx, ny, 1))

            # The initial pressure
            p = p_ref * (exn / cp) ** (cp / Rd)

            # The initial Montgomery potential
            mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
            mtg = np.zeros((nx, ny, nz), dtype=datatype)
            mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
            for k in range(1, nz):
                mtg[:, :, nz - k - 1] = (
                    mtg[:, :, nz - k] + dz * exn[:, :, nz - k]
                )

            # The initial geometric height of the isentropes
            h = np.zeros((nx, ny, nz + 1), dtype=datatype)
            h[:, :, -1] = self._grid.topography_height
            for k in range(0, nz):
                h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (
                    brunt_vaisala_initial ** 2 * z[nz - k - 1]
                )

            # The initial isentropic_prognostic density
            s = -1.0 / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

            # The initial water constituents
            if self._moist_on:
                # Set the initial relative humidity
                rhmax = 0.98
                kw = 10
                kc = 11
                k = np.arange(kc - kw + 1, kc + kw)
                rh1d = np.zeros(nz, dtype=datatype)
                rh1d[k] = (
                    rhmax * np.cos(np.abs(k - kc) * math.pi / (2.0 * kw)) ** 2
                )
                rh1d = rh1d[::-1]
                rh = np.tile(rh1d[np.newaxis, np.newaxis, :], (nx, ny, 1))

                # Compute the pressure and the temperature at the main levels
                p_ml = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
                theta = np.tile(
                    self._grid.z.values[np.newaxis, np.newaxis, :], (nx, ny, 1)
                )
                # T_ml  = theta * (p_ml / p_ref) ** (Rd / cp)
                theta_hl = np.tile(
                    self._grid.z_on_interface_levels.values[
                        np.newaxis, np.newaxis, :
                    ],
                    (nx, ny, 1),
                )
                T_ml = (
                    0.5
                    * (
                        theta_hl[:, :, :-1] * exn[:, :, :-1]
                        + theta_hl[:, :, 1:] * exn[:, :, 1:]
                    )
                    / cp
                )

                # Convert relative humidity to water vapor
                qv = utils_meteo.convert_relative_humidity_to_water_vapor(
                    "goff_gratch", p_ml, T_ml, rh
                )

                # Set the initial cloud water and precipitation water to zero
                qc = np.zeros((nx, ny, nz), dtype=datatype)
                qr = np.zeros((nx, ny, nz), dtype=datatype)

        #
        # Case 1
        #
        if initial_state_type == 1:
            # The initial velocity
            u = kwargs.get("x_velocity_initial", 10.0) * np.ones(
                (nx + 1, ny, nz), dtype=datatype
            )
            v = kwargs.get("y_velocity_initial", 0.0) * np.ones(
                (nx, ny + 1, nz), dtype=datatype
            )

            # The initial Exner function
            brunt_vaisala_initial = kwargs.get("brunt_vaisala_initial", 0.01)
            exn_col = np.zeros(nz + 1, dtype=datatype)
            exn_col[-1] = cp
            for k in range(0, nz):
                exn_col[nz - k - 1] = exn_col[nz - k] - dz * g ** 2 / (
                    brunt_vaisala_initial ** 2 * z[nz - k - 1] ** 2
                )
            exn = np.tile(exn_col[np.newaxis, np.newaxis, :], (nx, ny, 1))

            # The initial pressure
            p = p_ref * (exn / cp) ** (cp / Rd)

            # The initial Montgomery potential
            mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
            mtg = np.zeros((nx, ny, nz), dtype=datatype)
            mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
            for k in range(1, nz):
                mtg[:, :, nz - k - 1] = (
                    mtg[:, :, nz - k] + dz * exn[:, :, nz - k]
                )

            # The initial geometric height of the isentropes
            h = np.zeros((nx, ny, nz + 1), dtype=datatype)
            h[:, :, -1] = self._grid.topography_height
            for k in range(0, nz):
                h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (
                    brunt_vaisala_initial ** 2 * z[nz - k - 1]
                )

            # The initial isentropic_prognostic density
            s = -1.0 / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

            # The initial water constituents
            if self._moist_on:
                # Set the initial relative humidity
                rhmax = 0.98
                kw = 10
                kc = 11
                k = np.arange(kc - kw + 1, kc + kw)
                rh1d = np.zeros(nz, dtype=datatype)
                rh1d[k] = (
                    rhmax * np.cos(np.abs(k - kc) * math.pi / (2.0 * kw)) ** 2
                )
                rh1d = rh1d[::-1]
                rh = np.tile(rh1d[np.newaxis, np.newaxis, :], (nx, ny, 1))

                # Compute the pressure and the temperature at the main levels
                p_ml = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
                theta = np.tile(
                    self._grid.z.values[np.newaxis, np.newaxis, :], (nx, ny, 1)
                )
                T_ml = theta * (p_ml / p_ref) ** (Rd / cp)

                # Convert relative humidity to water vapor
                qv = utils_meteo.convert_relative_humidity_to_water_vapor(
                    "goff_gratch", p_ml, T_ml, rh
                )

                # Make the distribution of water vapor x-periodic
                x = np.tile(
                    self._grid.x.values[:, np.newaxis, np.newaxis], (1, ny, nz)
                )
                xl, xr = x[0, 0, 0], x[-1, 0, 0]
                qv = qv * (2.0 + np.sin(2.0 * math.pi * (x - xl) / (xr - xl)))

                # Set the initial cloud water and precipitation water to zero
                qc = np.zeros((nx, ny, nz), dtype=datatype)
                qr = np.zeros((nx, ny, nz), dtype=datatype)

        #
        # Case 2
        #
        if initial_state_type == 2:
            # The initial velocity
            u = kwargs.get("x_velocity_initial", 10.0) * np.ones(
                (nx + 1, ny, nz), dtype=datatype
            )
            v = kwargs.get("y_velocity_initial", 0.0) * np.ones(
                (nx, ny + 1, nz), dtype=datatype
            )

            # The initial pressure
            p = p_ref * (
                kwargs.get("temperature_initial", 250)
                / np.tile(z_hl[np.newaxis, np.newaxis, :], (nx, ny, 1))
            ) ** (cp / Rd)

            # The initial Exner function
            exn = cp * (p / p_ref) ** (Rd / cp)

            # The initial Montgomery potential
            mtg_s = z_hl[-1] * exn[:, :, -1] + g * topo
            mtg = np.zeros((nx, ny, nz), dtype=datatype)
            mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
            for k in range(1, nz):
                mtg[:, :, nz - k - 1] = (
                    mtg[:, :, nz - k] + dz * exn[:, :, nz - k]
                )

            # The initial geometric height of the isentropes
            h = np.zeros((nx, ny, nz + 1), dtype=datatype)
            h[:, :, -1] = self._grid.topography_height
            for k in range(0, nz):
                h[:, :, nz - k - 1] = (
                    h[:, :, nz - k]
                    - (p[:, :, nz - k - 1] - p[:, :, nz - k])
                    * Rd
                    / (cp * g)
                    * z_hl[nz - k]
                    * exn[:, :, nz - k]
                    / p[:, :, nz - k]
                )

            # The initial isentropic_prognostic density
            s = -1.0 / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

            # The initial water constituents
            if self._moist_on:
                qv = np.zeros((nx, ny, nz), dtype=datatype)
                qc = np.zeros((nx, ny, nz), dtype=datatype)
                qr = np.zeros((nx, ny, nz), dtype=datatype)

        # Assemble the initial state
        state = StateIsentropic(
            initial_time,
            self._grid,
            air_isentropic_density=s,
            x_velocity=u,
            y_velocity=v,
            air_pressure_on_interface_levels=p,
            exner_function_on_interface_levels=exn,
            montgomery_potential=mtg,
            height_on_interface_levels=h,
        )

        # Diagnose the air density and temperature
        state.extend(self._diagnostic.get_air_density(state)),
        state.extend(self._diagnostic.get_air_temperature(state))

        if self._moist_on:
            # Add the mass fraction of each water component
            state.add_variables(
                initial_time,
                mass_fraction_of_water_vapor_in_air=qv,
                mass_fraction_of_cloud_liquid_water_in_air=qc,
                mass_fraction_of_precipitation_water_in_air=qr,
            )

        return state

    def _step_dry(self, dt, state, diagnostics, tendencies):
        """
        Method advancing the dry isentropic_prognostic state by a single time step.

        Parameters
        ----------
        dt : obj
                :class:`datetime.timedelta` representing the time step.
        state :obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
                It should contain the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
                * montgomery_potential (unstaggered).

        diagnostics : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` storing diagnostics.
                For the time being, this is not actually used.
        tendencies : `obj`, optional
                :class:`~tasmania.storages.grid_data.GridData` storing tendencies.
                For the time being, this is not actually used.

        Return
        ------
        state_new : obj
                :class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the state at the next time level.
                It contains the following variables:

                * air_isentropic_density (unstaggered);
                * x_velocity (:math:`x`-staggered);
                * y_velocity (:math:`y`-staggered);
                * air_pressure_on_interface_levels (:math:`z`-staggered);
                * exner_function_on_interface_levels (:math:`z`-staggered);
                * montgomery_potential (unstaggered);
                * height_on_interface_levels (:math:`z`-staggered).

        diagnostics_out : obj
                Empty :class:`~tasmania.tasmania.storages.grid_data.GridData`, as no diagnostics are computed.
        """
        # Initialize the empty GridData to return
        time_now = utils.convert_datetime64_to_datetime(
            state["air_isentropic_density"].coords["time"].values[0]
        )
        diagnostics_out = GridData(time_now + dt, self._grid)

        # If either damping or smoothing is enabled: extract the prognostic model variables
        if self._damp_on or self._smooth_on:
            s_now = state["air_isentropic_density"].values[:, :, :, 0]
            u_now = state["x_velocity"].values[:, :, :, 0]
            v_now = state["y_velocity"].values[:, :, :, 0]

        # Perform the prognostic step
        state_new = self._prognostic.step_neglecting_vertical_advection(
            dt, state, diagnostics=diagnostics, tendencies=tendencies
        )

        if self._damp_on:
            # If this is the first call to the entry-point method: set the reference state
            if not hasattr(self, "_s_ref"):
                self._s_ref = np.copy(s_now)
                self._u_ref = np.copy(u_now)
                self._v_ref = np.copy(v_now)

            # Extract the prognostic model variables
            s_new = state_new["air_isentropic_density"].values[:, :, :, 0]
            u_new = state_new["x_velocity"].values[:, :, :, 0]
            v_new = state_new["y_velocity"].values[:, :, :, 0]

            # Apply vertical damping
            s_new[:, :, :] = self._damper_unstg.apply(
                dt, s_now, s_new, self._s_ref
            )
            u_new[:, :, :] = self._damper_stg_x.apply(
                dt, u_now, u_new, self._u_ref
            )
            v_new[:, :, :] = self._damper_stg_y.apply(
                dt, v_now, v_new, self._v_ref
            )

        if self._smooth_on:
            if not self._damp_on:
                # Extract the prognostic model variables
                s_new = state_new["air_isentropic_density"].values[:, :, :, 0]
                u_new = state_new["x_velocity"].values[:, :, :, 0]
                v_new = state_new["y_velocity"].values[:, :, :, 0]

            # Apply horizontal smoothing
            s_new[:, :, :] = self._smoother_s.apply(s_new)
            u_new[:, :, :] = self._smoother_u.apply(u_new)
            v_new[:, :, :] = self._smoother_v.apply(v_new)

            # Apply horizontal boundary conditions
            self._boundary.apply(s_new, s_now)
            self._boundary.apply(u_new, u_now)
            self._boundary.apply(v_new, v_now)

        # Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
        p_ = (
            state["air_pressure"]
            if state["air_pressure"] is not None
            else state["air_pressure_on_interface_levels"]
        )
        state_new.extend(
            self._diagnostic.get_diagnostic_variables(
                state_new, p_.values[0, 0, 0, 0]
            )
        )

        return state_new, diagnostics_out

    def _step_moist(self, dt, state, diagnostics, tendencies):
        """
		Method advancing the moist isentropic_prognostic state by a single time step.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		state :obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		diagnostics : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` storing diagnostics, namely:

			* change_over_time_in_air_potential_temperature (unstaggered, required only if coupling \
				between physics and dynamics is switched on);
			* accumulated_precipitation (unstaggered).

		tendencies : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` possibly storing tendencies.
			For the time being, this is not actually used.

		Return
		------
		state_new : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the state at the next time level.
			It contains the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* air_pressure_on_interface_levels (:math:`z`-staggered);
			* exner_function_on_interface_levels (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height_on_interface_levels (:math:`z`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered);
			* air_density (unstaggered, only if cloud microphysics is switched on);
			* air_temperature (unstaggered, only if cloud microphysics is switched on).

		diagnostics_out : obj
			:class:`~tasmania.tasmania.storages.grid_data.GridData` collecting output diagnostics, namely:

			* precipitation (unstaggered, only if rain sedimentation is switched on);
			* accumulated_precipitation (unstaggered, only if rain sedimentation is switched on).
		"""
        # Initialize the GridData to return
        time_now = utils.convert_datetime64_to_datetime(
            state["air_isentropic_density"].coords["time"].values[0]
        )
        diagnostics_out = GridData(
            time_now + dt,
            self._grid,
            precipitation=np.zeros(
                (self._grid.nx, self._grid.ny), dtype=datatype
            ),
            accumulated_precipitation=np.zeros(
                (self._grid.nx, self._grid.ny), dtype=datatype
            ),
        )

        # If either damping or smoothing is enabled: extract the prognostic model variables
        if self._damp_on or self._smooth_on:
            s_now = state["air_isentropic_density"].values[:, :, :, 0]
            u_now = state["x_velocity"].values[:, :, :, 0]
            v_now = state["y_velocity"].values[:, :, :, 0]
            qv_now = state["mass_fraction_of_water_vapor_in_air"].values[
                :, :, :, 0
            ]
            qc_now = state[
                "mass_fraction_of_cloud_liquid_water_in_air"
            ].values[:, :, :, 0]
            qr_now = state[
                "mass_fraction_of_precipitation_water_in_air"
            ].values[:, :, :, 0]

        # Perform the prognostic step, neglecting the vertical advection
        state_new = self._prognostic.step_neglecting_vertical_advection(
            dt, state, diagnostics=diagnostics, tendencies=tendencies
        )

        if self._physics_dynamics_coupling_on:
            # Couple physics with dynamics
            state_new_ = self._prognostic.step_coupling_physics_with_dynamics(
                dt, state, state_new, diagnostics
            )

            # Update the output state
            state_new.update(state_new_)

        if self._damp_on:
            # If this is the first call to the entry-point method: set the reference state
            if not hasattr(self, "_s_ref"):
                self._s_ref = np.copy(s_now)
                self._u_ref = np.copy(u_now)
                self._v_ref = np.copy(v_now)
                self._qv_ref = np.copy(qv_now)
                self._qc_ref = np.copy(qc_now)
                self._qr_ref = np.copy(qr_now)

            # Extract the prognostic model variables
            s_new = state_new["air_isentropic_density"].values[:, :, :, 0]
            u_new = state_new["x_velocity"].values[:, :, :, 0]
            v_new = state_new["y_velocity"].values[:, :, :, 0]
            qv_new = state_new["mass_fraction_of_water_vapor_in_air"].values[
                :, :, :, 0
            ]
            qc_new = state_new[
                "mass_fraction_of_cloud_liquid_water_in_air"
            ].values[:, :, :, 0]
            qr_new = state_new[
                "mass_fraction_of_precipitation_water_in_air"
            ].values[:, :, :, 0]

            # Apply vertical damping
            s_new[:, :, :] = self._damper_unstg.apply(
                dt, s_now, s_new, self._s_ref
            )
            u_new[:, :, :] = self._damper_stg_x.apply(
                dt, u_now, u_new, self._u_ref
            )
            v_new[:, :, :] = self._damper_stg_y.apply(
                dt, v_now, v_new, self._v_ref
            )
            qv_new[:, :, :] = self._damper_unstg.apply(
                dt, qv_now, qv_new, self._qv_ref
            )
            qc_new[:, :, :] = self._damper_unstg.apply(
                dt, qc_now, qc_new, self._qc_ref
            )
            qr_new[:, :, :] = self._damper_unstg.apply(
                dt, qr_now, qr_new, self._qr_ref
            )

        if self._smooth_on:
            if not self._damp_on:
                # Extract the dry prognostic model variables
                s_new = state_new["air_isentropic_density"].values[:, :, :, 0]
                u_new = state_new["x_velocity"].values[:, :, :, 0]
                v_new = state_new["y_velocity"].values[:, :, :, 0]

            # Apply horizontal smoothing
            s_new[:, :, :] = self._smoother_s.apply(s_new)
            u_new[:, :, :] = self._smoother_u.apply(u_new)
            v_new[:, :, :] = self._smoother_v.apply(v_new)

            # Apply horizontal boundary conditions
            self._boundary.apply(s_new, s_now)
            self._boundary.apply(u_new, u_now)
            self._boundary.apply(v_new, v_now)

        if self._smooth_moist_on:
            if not self._damp_on:
                # Extract the moist prognostic model variables
                qv_new = state_new[
                    "mass_fraction_of_water_vapor_in_air"
                ].values[:, :, :, 0]
                qc_new = state_new[
                    "mass_fraction_of_cloud_liquid_water_in_air"
                ].values[:, :, :, 0]
                qr_new = state_new[
                    "mass_fraction_of_precipitation_water_in_air"
                ].values[:, :, :, 0]

            # Apply horizontal smoothing
            qv_new[:, :, :] = self._smoother_q.apply(qv_new)
            qc_new[:, :, :] = self._smoother_q.apply(qc_new)
            qr_new[:, :, :] = self._smoother_q.apply(qr_new)

            # Apply horizontal boundary conditions
            self._boundary.apply(qv_new, qv_now)
            self._boundary.apply(qc_new, qc_now)
            self._boundary.apply(qr_new, qr_now)

        # Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
        p_ = (
            state["air_pressure"]
            if state["air_pressure"] is not None
            else state["air_pressure_on_interface_levels"]
        )
        state_new.extend(
            self._diagnostic.get_diagnostic_variables(
                state_new, p_.values[0, 0, 0, 0]
            )
        )

        if self.microphysics is not None:
            # Diagnose the density
            state_new.extend(self._diagnostic.get_air_density(state_new))

            # Diagnose the temperature
            state_new.extend(self._diagnostic.get_air_temperature(state_new))

        if self._sedimentation_on:
            qr = state["mass_fraction_of_precipitation_water_in_air"].values[
                :, :, :, 0
            ]
            qr_new = state_new[
                "mass_fraction_of_precipitation_water_in_air"
            ].values[:, :, :, 0]

            if np.any(qr > 0.0) or np.any(qr_new > 0.0):
                # Integrate rain sedimentation flux
                (
                    state_new_,
                    diagnostics_out_,
                ) = self._prognostic.step_integrating_sedimentation_flux(
                    dt, state, state_new, diagnostics
                )

                # Update the output state and the output diagnostics
                state_new.update(state_new_)
                diagnostics_out.update(diagnostics_out_)

        return state_new, diagnostics_out
