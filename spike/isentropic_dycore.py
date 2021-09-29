# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
            mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

        # The initial geometric height of the isentropes
        h = np.zeros((nx, ny, nz + 1), dtype=datatype)
        h[:, :, -1] = self._grid.topography_height
        for k in range(0, nz):
            h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (
                brunt_vaisala_initial ** 2 * z[nz - k - 1]
            )

        # The initial isentropic_prognostic density
        s = -1.0 / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

        # The initial momentums
        U = s * kwargs.get("x_velocity_initial", 10.0)
        V = s * kwargs.get("y_velocity_initial", 0.0)

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
            mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

        # The initial geometric height of the isentropes
        h = np.zeros((nx, ny, nz + 1), dtype=datatype)
        h[:, :, -1] = self._grid.topography_height
        for k in range(0, nz):
            h[:, :, nz - k - 1] = h[:, :, nz - k] + dz * g / (
                brunt_vaisala_initial ** 2 * z[nz - k - 1]
            )

        # The initial isentropic_prognostic density
        s = -1.0 / g * (p[:, :, :-1] - p[:, :, 1:]) / dz

        # The initial momentums
        U = s * kwargs.get("x_velocity_initial", 10.0)
        V = s * kwargs.get("y_velocity_initial", 0.0)

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
            mtg[:, :, nz - k - 1] = mtg[:, :, nz - k] + dz * exn[:, :, nz - k]

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

        # The initial momentums
        U = s * kwargs.get("x_velocity_initial", 10.0)
        V = s * kwargs.get("y_velocity_initial", 0.0)

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
        x_momentum_isentropic=U,
        y_velocity=v,
        y_momentum_isentropic=V,
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


class SUSHomogeneousIsentropicDynamicalCore(SequentialSplittingDynamicalCore):
    """
    This class inherits :class:`~tasmania.dynamics.dycore.SequentialSplittingDynamicalCore`
    to implement the three-dimensional (moist) isentropic_prognostic homogeneous dynamical core.
    Here, _homogeneous_ means that the pressure gradient terms, i.e., the terms
    involving the gradient of the Montgomery potential, are not included in the dynamics,
    but rather parameterized.
    The class supports different numerical schemes to carry out the prognostic
    steps of the dynamical core, and different types of lateral boundary conditions.
    The conservative form of the governing equations is used.
    The sequential-update splitting method is pursued.
    """

    def __init__(
        self,
        grid,
        moist,
        time_integration_scheme,
        horizontal_flux_scheme,
        horizontal_boundary_type,
        damp=True,
        damp_type="rayleigh",
        damp_depth=15,
        damp_max=0.0002,
        smooth=True,
        smooth_type="first_order",
        smooth_damp_depth=10,
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_moist=False,
        smooth_moist_type="first_order",
        smooth_moist_damp_depth=10,
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        backend=gt.mode.NUMPY,
        dtype=datatype,
    ):
        """
        Constructor.

        Parameters
        ----------
        grid : grid
                :class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
        moist : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        time_integration_scheme : str
                String specifying the time stepping method to implement.
                See :class:`~tasmania.dynamics.homogeneous_isentropic_prognostic.HomogeneousIsentropicPrognostic`
                for all available options.
        horizontal_flux_scheme : str
                String specifying the numerical horizontal flux to use.
                See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousIsentropicFlux`
                for all available options.
        horizontal_boundary_type : str
                String specifying the horizontal boundary conditions.
                See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
                for all available options.
        damp : `bool`, optional
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
        smooth : `bool`, optional
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
        smooth_moist : `bool`, optional
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
        backend : `obj`, optional
                :class:`gridtools.mode` specifying the backend for the GT4Py stencils
                implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
        dtype : `obj`, optional
                Instance of :class:`numpy.dtype` specifying the data type for
                any :class:`numpy.ndarray` used within this class.
                Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
                if :obj:`~tasmania.namelist.datatype` is not defined.
        """
        # Keep track of the input parameters
        self._damp = damp
        self._smooth = smooth
        self._smooth_moist = smooth_moist
        self._dtype = dtype

        # Instantiate the class taking care of the boundary conditions
        self._boundary = HorizontalBoundary.factory(
            horizontal_boundary_type, grid
        )

        # Instantiate the classes implementing the prognostic part of the dycore
        self._prognostic_x = HomogeneousIsentropicPrognostic.factory(
            time_integration_scheme,
            "x",
            grid,
            moist,
            self._boundary,
            horizontal_flux_scheme,
            backend,
            dtype,
        )
        self._prognostic_y = HomogeneousIsentropicPrognostic.factory(
            time_integration_scheme,
            "y",
            grid,
            moist,
            self._boundary,
            horizontal_flux_scheme,
            backend,
            dtype,
        )

        # Instantiate the class in charge of applying vertical damping
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        if damp:
            self._damper = VerticalDamping.factory(
                damp_type,
                (nx, ny, nz),
                grid,
                damp_depth,
                damp_max,
                backend,
                dtype,
            )

        # Instantiate the classes in charge of applying numerical smoothing
        if smooth:
            self._smoother = HorizontalSmoothing.factory(
                smooth_type,
                (nx, ny, nz),
                grid,
                smooth_damp_depth,
                smooth_coeff,
                smooth_coeff_max,
                backend,
                dtype,
            )
            if moist and smooth_moist:
                self._smoother_moist = HorizontalSmoothing.factory(
                    smooth_moist_type,
                    (nx, ny, nz),
                    grid,
                    smooth_moist_damp_depth,
                    smooth_moist_coeff,
                    smooth_moist_coeff_max,
                    backend,
                    dtype,
                )

        # Instantiate the class in charge of diagnosing the velocity components
        self._velocity_components = HorizontalVelocity(grid, backend, dtype)

        # Instantiate the class in charge of diagnosing the mass fraction and
        # isentropic_prognostic density of the water constituents
        if moist:
            self._water_constituent = WaterConstituent(grid, backend, dtype)

        # Set the pointer to the private method implementing each stage
        self._array_call = (
            self._array_call_dry if not moist else self._array_call_moist
        )

        # Call parent constructor
        super().__init__(grid, moist)

    @property
    def input_properties(self):
        dims = (
            self._grid.x.dims[0],
            self._grid.y.dims[0],
            self._grid.z.dims[0],
        )
        dims_stg_x = (
            self._grid.x_at_u_locations.dims[0],
            self._grid.y.dims[0],
            self._grid.z.dims[0],
        )
        dims_stg_y = (
            self._grid.x.dims[0],
            self._grid.y_at_v_locations.dims[0],
            self._grid.z.dims[0],
        )

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "x_velocity_at_u_locations": {
                "dims": dims_stg_x,
                "units": "m s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_velocity_at_v_locations": {
                "dims": dims_stg_y,
                "units": "m s^-1",
            },
        }

        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }

        return return_dict

    @property
    def output_properties(self):
        dims = (
            self._grid.x.dims[0],
            self._grid.y.dims[0],
            self._grid.z.dims[0],
        )
        dims_stg_x = (
            self._grid.x_at_u_locations.dims[0],
            self._grid.y.dims[0],
            self._grid.z.dims[0],
        )
        dims_stg_y = (
            self._grid.x.dims[0],
            self._grid.y_at_v_locations.dims[0],
            self._grid.z.dims[0],
        )

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "x_velocity_at_u_locations": {
                "dims": dims_stg_x,
                "units": "m s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_velocity_at_v_locations": {
                "dims": dims_stg_y,
                "units": "m s^-1",
            },
        }

        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }

        return return_dict

    def array_call(self, raw_state, timestep):
        return self._array_call(raw_state, timestep)

    def _array_call_dry(self, raw_state, timestep):
        """
        Perform a timestep of the dry dynamical core.
        """
        # Shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dtype = self._dtype

        # Advance the solution
        raw_state_new = self._prognostic_call(raw_state, timestep)

        damped = False
        if self._damp:
            damped = True

            # If this is the first call to the entry-point method,
            # set the reference state...
            if not hasattr(self, "_s_ref"):
                self._s_ref = np.copy(raw_state["air_isentropic_density"])
                self._su_ref = np.copy(raw_state["x_momentum_isentropic"])
                self._sv_ref = np.copy(raw_state["y_momentum_isentropic"])

            # ...and allocate memory to store damped fields
            if not hasattr(self, "_s_damped"):
                self._s_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)

            # Extract the current prognostic model variables
            s_now_ = raw_state["air_isentropic_density"]
            su_now_ = raw_state["x_momentum_isentropic"]
            sv_now_ = raw_state["y_momentum_isentropic"]

            # Extract the stepped prognostic model variables
            s_new_ = raw_state_new["air_isentropic_density"]
            su_new_ = raw_state_new["x_momentum_isentropic"]
            sv_new_ = raw_state_new["y_momentum_isentropic"]

            # Apply vertical damping
            self._damper(timestep, s_now_, s_new_, self._s_ref, self._s_damped)
            self._damper(
                timestep, su_now_, su_new_, self._su_ref, self._su_damped
            )
            self._damper(
                timestep, sv_now_, sv_new_, self._sv_ref, self._sv_damped
            )

        # Properly set pointers to current solution
        s_new = (
            self._s_damped
            if damped
            else raw_state_new["air_isentropic_density"]
        )
        su_new = (
            self._su_damped
            if damped
            else raw_state_new["x_momentum_isentropic"]
        )
        sv_new = (
            self._sv_damped
            if damped
            else raw_state_new["y_momentum_isentropic"]
        )

        smoothed = False
        if self._smooth:
            smoothed = True

            # If this is the first call to the entry-point method,
            # allocate memory to store smoothed fields
            if not hasattr(self, "_s_smoothed"):
                self._s_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

            # Apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)

            # Apply horizontal boundary conditions
            self._boundary.enforce(self._s_smoothed, s_new)
            self._boundary.enforce(self._su_smoothed, su_new)
            self._boundary.enforce(self._sv_smoothed, sv_new)

        # Properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new

        # Diagnose the velocity components
        u_out, v_out = self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out
        )
        self._boundary.set_outermost_layers_x(
            u_out, raw_state["x_velocity_at_u_locations"]
        )
        self._boundary.set_outermost_layers_y(
            v_out, raw_state["y_velocity_at_v_locations"]
        )

        # Instantiate the output state
        raw_state_out = {
            "air_isentropic_density": s_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": v_out,
        }

        return raw_state_out

    def _array_call_moist(self, raw_state, timestep):
        """
        Perform a timestep of the moist dynamical core.
        """
        # Shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dtype = self._dtype

        # Allocate memory to store the isentropic_prognostic density of all water constituents
        if not hasattr(self, "_sqv_now"):
            self._sqv_now = np.zeros((nx, ny, nz), dtype=dtype)
            self._sqc_now = np.zeros((nx, ny, nz), dtype=dtype)
            self._sqr_now = np.zeros((nx, ny, nz), dtype=dtype)

        # Diagnosed the isentropic_prognostic density of all water constituents
        s_now = raw_state["air_isentropic_density"]
        qv_now = raw_state["mass_fraction_of_water_vapor_in_air"]
        qc_now = raw_state["mass_fraction_of_cloud_liquid_water_in_air"]
        qr_now = raw_state["mass_fraction_of_precipitation_water_in_air"]
        self._water_constituent.get_density_of_water_constituent(
            s_now, qv_now, self._sqv_now
        )
        self._water_constituent.get_density_of_water_constituent(
            s_now, qc_now, self._sqc_now
        )
        self._water_constituent.get_density_of_water_constituent(
            s_now, qr_now, self._sqr_now
        )
        raw_state["isentropic_density_of_water_vapor"] = self._sqv_now
        raw_state["isentropic_density_of_cloud_liquid_water"] = self._sqc_now
        raw_state["isentropic_density_of_precipitation_water"] = self._sqr_now

        # Advance the solution
        raw_state_new = self._prognostic_call(raw_state, timestep)

        damped = False
        if self._damp:
            damped = True

            # If this is the first call to the entry-point method,
            # set the reference state...
            if not hasattr(self, "_s_ref"):
                self._s_ref = np.copy(raw_state["air_isentropic_density"])
                self._su_ref = np.copy(raw_state["x_momentum_isentropic"])
                self._sv_ref = np.copy(raw_state["y_momentum_isentropic"])
                self._sqv_ref = np.copy(
                    raw_state["isentropic_density_of_water_vapor"]
                )
                self._sqc_ref = np.copy(
                    raw_state["isentropic_density_of_cloud_liquid_water"]
                )
                self._sqr_ref = np.copy(
                    raw_state["isentropic_density_of_precipitation_water"]
                )

            # ...and allocate memory to store damped fields
            if not hasattr(self, "_s_damped"):
                self._s_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._sqv_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._sqc_damped = np.zeros((nx, ny, nz), dtype=dtype)
                self._sqr_damped = np.zeros((nx, ny, nz), dtype=dtype)

            # Extract the current prognostic model variables
            s_now_ = raw_state["air_isentropic_density"]
            su_now_ = raw_state["x_momentum_isentropic"]
            sv_now_ = raw_state["y_momentum_isentropic"]
            sqv_now_ = raw_state["isentropic_density_of_water_vapor"]
            sqc_now_ = raw_state["isentropic_density_of_cloud_liquid_water"]
            sqr_now_ = raw_state["isentropic_density_of_precipitation_water"]

            # Extract the stepped prognostic model variables
            s_new_ = raw_state_new["air_isentropic_density"]
            su_new_ = raw_state_new["x_momentum_isentropic"]
            sv_new_ = raw_state_new["y_momentum_isentropic"]
            sqv_new_ = raw_state_new["isentropic_density_of_water_vapor"]
            sqc_new_ = raw_state_new[
                "isentropic_density_of_cloud_liquid_water"
            ]
            sqr_new_ = raw_state_new[
                "isentropic_density_of_precipitation_water"
            ]

            # Apply vertical damping
            self._damper(timestep, s_now_, s_new_, self._s_ref, self._s_damped)
            self._damper(
                timestep, su_now_, su_new_, self._su_ref, self._su_damped
            )
            self._damper(
                timestep, sv_now_, sv_new_, self._sv_ref, self._sv_damped
            )
            self._damper(
                timestep, sqv_now_, sqv_new_, self._sqv_ref, self._sqv_damped
            )
            self._damper(
                timestep, sqc_now_, sqc_new_, self._sqc_ref, self._sqc_damped
            )
            self._damper(
                timestep, sqr_now_, sqr_new_, self._sqr_ref, self._sqr_damped
            )

        # Properly set pointers to current solution
        s_new = (
            self._s_damped
            if damped
            else raw_state_new["air_isentropic_density"]
        )
        su_new = (
            self._su_damped
            if damped
            else raw_state_new["x_momentum_isentropic"]
        )
        sv_new = (
            self._sv_damped
            if damped
            else raw_state_new["y_momentum_isentropic"]
        )
        sqv_new = (
            self._sqv_damped
            if damped
            else raw_state_new["isentropic_density_of_water_vapor"]
        )
        sqc_new = (
            self._sqc_damped
            if damped
            else raw_state_new["isentropic_density_of_cloud_liquid_water"]
        )
        sqr_new = (
            self._sqr_damped
            if damped
            else raw_state_new["isentropic_density_of_precipitation_water"]
        )

        smoothed = False
        if self._smooth:
            smoothed = True

            # If this is the first call to the entry-point method,
            # allocate memory to store smoothed fields
            if not hasattr(self, "_s_smoothed"):
                self._s_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

            # Apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)

            # Apply horizontal boundary conditions
            self._boundary.enforce(self._s_smoothed, s_new)
            self._boundary.enforce(self._su_smoothed, su_new)
            self._boundary.enforce(self._sv_smoothed, sv_new)

        # Properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new

        smoothed_moist = False
        if self._smooth_moist:
            smoothed_moist = True

            # If this is the first call to the entry-point method,
            # allocate memory to store smoothed fields
            if not hasattr(self, "_sqv_smoothed"):
                self._sqv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._sqc_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
                self._sqr_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

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
        u_out, v_out = self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out
        )
        self._boundary.set_outermost_layers_x(
            u_out, raw_state["x_velocity_at_u_locations"]
        )
        self._boundary.set_outermost_layers_y(
            v_out, raw_state["y_velocity_at_v_locations"]
        )

        # Allocate memory to store the isentropic_prognostic density of all water constituents
        if not hasattr(self, "_qv_out"):
            self._qv_out = np.zeros((nx, ny, nz), dtype=dtype)
            self._qc_out = np.zeros((nx, ny, nz), dtype=dtype)
            self._qr_out = np.zeros((nx, ny, nz), dtype=dtype)

        # Diagnose the mass fraction of all water constituents
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_out, sqv_out, self._qv_out, clipping=True
        )
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_out, sqc_out, self._qc_out, clipping=True
        )
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_out, sqr_out, self._qr_out, clipping=True
        )

        # Instantiate the output state
        raw_state_out = {
            "air_isentropic_density": s_out,
            "mass_fraction_of_water_vapor_in_air": self._qv_out,
            "mass_fraction_of_cloud_liquid_water_in_air": self._qc_out,
            "mass_fraction_of_precipitation_water_in_air": self._qr_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": v_out,
        }

        return raw_state_out

    def _prognostic_call(self, raw_state, timestep):
        # Shorthands
        ns = self._prognostic_x.stages

        # Initialize the output state
        raw_state_new = {}
        raw_state_new.update(raw_state)

        # Advance the solution considering only the x-advection
        for k in range(ns):
            raw_state_tmp = self._prognostic_x(k, timestep, raw_state_new)
            raw_state_new.update(raw_state_tmp)

            # Diagnose the velocity components
            u_new, v_new = self._velocity_components.get_velocity_components(
                raw_state_new["air_isentropic_density"],
                raw_state_new["x_momentum_isentropic"],
                raw_state_new["y_momentum_isentropic"],
            )
            self._boundary.set_outermost_layers_x(
                u_new, raw_state["x_velocity_at_u_locations"]
            )
            self._boundary.set_outermost_layers_y(
                v_new, raw_state["y_velocity_at_v_locations"]
            )

            raw_state_new["x_velocity_at_u_locations"] = u_new
            raw_state_new["y_velocity_at_v_locations"] = v_new

        # Advance the solution considering only the y-advection
        for k in range(ns):
            raw_state_tmp = self._prognostic_y(k, timestep, raw_state_new)
            raw_state_new.update(raw_state_tmp)

            # Diagnose the velocity components
            u_new, v_new = self._velocity_components.get_velocity_components(
                raw_state_new["air_isentropic_density"],
                raw_state_new["x_momentum_isentropic"],
                raw_state_new["y_momentum_isentropic"],
            )
            self._boundary.set_outermost_layers_x(
                u_new, raw_state["x_velocity_at_u_locations"]
            )
            self._boundary.set_outermost_layers_y(
                v_new, raw_state["y_velocity_at_v_locations"]
            )

            raw_state_new["x_velocity_at_u_locations"] = u_new
            raw_state_new["y_velocity_at_v_locations"] = v_new

        return raw_state_new


class SSUSHomogeneousIsentropicDynamicalCore(
    SUSHomogeneousIsentropicDynamicalCore
):
    """
    This class inherits
    :class:`~tasmania.dynamics.homogeneous_isentropic_dycore.SUSHomogeneousIsentropicDynamicalCore`
    to implement the three-dimensional (moist) isentropic_prognostic homogeneous dynamical core.
    Here, _homogeneous_ means that the pressure gradient terms, i.e., the terms
    involving the gradient of the Montgomery potential, are not included in the dynamics,
    but rather parameterized.
    The class supports different numerical schemes to carry out the prognostic
    steps of the dynamical core, and different types of lateral boundary conditions.
    The conservative form of the governing equations is used.
    The symmetrized sequential-update splitting method is pursued.
    """

    def __init__(
        self,
        grid,
        moist,
        time_integration_scheme,
        horizontal_flux_scheme,
        horizontal_boundary_type,
        damp=True,
        damp_type="rayleigh",
        damp_depth=15,
        damp_max=0.0002,
        smooth=True,
        smooth_type="first_order",
        smooth_damp_depth=10,
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_moist=False,
        smooth_moist_type="first_order",
        smooth_moist_damp_depth=10,
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        backend=gt.mode.NUMPY,
        dtype=datatype,
    ):
        """
        Constructor.

        Parameters
        ----------
        grid : grid
                :class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
        moist : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        time_integration_scheme : str
                String specifying the time stepping method to implement.
                See :class:`~tasmania.dynamics.homogeneous_isentropic_prognostic.HomogeneousIsentropicPrognostic`
                for all available options.
        horizontal_flux_scheme : str
                String specifying the numerical horizontal flux to use.
                See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousIsentropicFlux`
                for all available options.
        horizontal_boundary_type : str
                String specifying the horizontal boundary conditions.
                See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
                for all available options.
        damp : `bool`, optional
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
        smooth : `bool`, optional
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
        smooth_moist : `bool`, optional
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
        backend : `obj`, optional
                :class:`gridtools.mode` specifying the backend for the GT4Py stencils
                implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
        dtype : `obj`, optional
                Instance of :class:`numpy.dtype` specifying the data type for
                any :class:`numpy.ndarray` used within this class.
                Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
                if :obj:`~tasmania.namelist.datatype` is not defined.
        """
        super().__init__(
            grid,
            moist,
            time_integration_scheme,
            horizontal_flux_scheme,
            horizontal_boundary_type,
            damp,
            damp_type,
            damp_depth,
            damp_max,
            smooth,
            smooth_type,
            smooth_damp_depth,
            smooth_coeff,
            smooth_coeff_max,
            smooth_moist,
            smooth_moist_type,
            smooth_moist_damp_depth,
            smooth_moist_coeff,
            smooth_moist_coeff_max,
            backend,
            dtype,
        )

    def _prognostic_call(self, raw_state, timestep):
        # Shorthands
        ns = self._prognostic_x.stages

        # Initialize the output state
        raw_state_new = {}
        raw_state_new.update(raw_state)

        # Advance the solution considering only the y-advection
        for k in range(ns):
            raw_state_tmp = self._prognostic_y(
                k, 0.5 * timestep, raw_state_new
            )
            raw_state_new.update(raw_state_tmp)

            # Diagnose the velocity components
            u_new, v_new = self._velocity_components.get_velocity_components(
                raw_state_new["air_isentropic_density"],
                raw_state_new["x_momentum_isentropic"],
                raw_state_new["y_momentum_isentropic"],
            )
            self._boundary.set_outermost_layers_x(
                u_new, raw_state["x_velocity_at_u_locations"]
            )
            self._boundary.set_outermost_layers_y(
                v_new, raw_state["y_velocity_at_v_locations"]
            )

            raw_state_new["x_velocity_at_u_locations"] = u_new
            raw_state_new["y_velocity_at_v_locations"] = v_new

        # Advance the solution considering only the x-advection
        for k in range(ns):
            raw_state_tmp = self._prognostic_x(k, timestep, raw_state_new)
            raw_state_new.update(raw_state_tmp)

            # Diagnose the velocity components
            u_new, v_new = self._velocity_components.get_velocity_components(
                raw_state_new["air_isentropic_density"],
                raw_state_new["x_momentum_isentropic"],
                raw_state_new["y_momentum_isentropic"],
            )
            self._boundary.set_outermost_layers_x(
                u_new, raw_state["x_velocity_at_u_locations"]
            )
            self._boundary.set_outermost_layers_y(
                v_new, raw_state["y_velocity_at_v_locations"]
            )

            raw_state_new["x_velocity_at_u_locations"] = u_new
            raw_state_new["y_velocity_at_v_locations"] = v_new

        # Advance the solution considering only the y-advection
        for k in range(ns):
            raw_state_tmp = self._prognostic_y(
                k, 0.5 * timestep, raw_state_new
            )
            raw_state_new.update(raw_state_tmp)

            # Diagnose the velocity components
            u_new, v_new = self._velocity_components.get_velocity_components(
                raw_state_new["air_isentropic_density"],
                raw_state_new["x_momentum_isentropic"],
                raw_state_new["y_momentum_isentropic"],
            )
            self._boundary.set_outermost_layers_x(
                u_new, raw_state["x_velocity_at_u_locations"]
            )
            self._boundary.set_outermost_layers_y(
                v_new, raw_state["y_velocity_at_v_locations"]
            )

            raw_state_new["x_velocity_at_u_locations"] = u_new
            raw_state_new["y_velocity_at_v_locations"] = v_new

        return raw_state_new
