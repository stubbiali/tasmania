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
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple, Union

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


# convenient shortcuts
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicDynamicalCore(DynamicalCore):
    """
    The three-dimensional (moist) isentropic dynamical core. Note that only
    the pressure gradient is included in the so-called *dynamics*. Any other
    large-scale process (e.g., vertical advection, Coriolis acceleration) might
    be included in the model only via physical parameterizations.
    The conservative form of the governing equations is used.
    """

    def __init__(
        self,
        domain: "Domain",
        intermediate_tendencies: Optional[taz_types.tendency_component_t] = None,
        intermediate_diagnostics: Optional[
            Union[taz_types.diagnostic_component_t, taz_types.tendency_component_t]
        ] = None,
        substeps: int = 0,
        fast_tendencies: Optional[taz_types.tendency_component_t] = None,
        fast_diagnostics: Optional[taz_types.diagnostic_component_t] = None,
        moist: bool = False,
        time_integration_scheme: str = "forward_euler_si",
        horizontal_flux_scheme: str = "upwind",
        time_integration_properties: Optional[taz_types.options_dict_t] = None,
        damp: bool = True,
        damp_at_every_stage: bool = True,
        damp_type: str = "rayleigh",
        damp_depth: int = 15,
        damp_max: float = 0.0002,
        smooth: bool = True,
        smooth_at_every_stage: bool = True,
        smooth_type: str = "first_order",
        smooth_coeff: float = 0.03,
        smooth_coeff_max: float = 0.24,
        smooth_damp_depth: int = 10,
        smooth_moist: bool = False,
        smooth_moist_at_every_stage: bool = True,
        smooth_moist_type: str = "first_order",
        smooth_moist_coeff: float = 0.03,
        smooth_moist_coeff_max: float = 0.24,
        smooth_moist_damp_depth: int = 10,
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        intermediate_tendencies : `obj`, optional
            An instance of either

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            calculating the intermediate physical tendencies.
            Here, *intermediate* refers to the fact that these physical
            packages are called *before* each stage of the dynamical core
            to calculate the physical tendencies.
        intermediate_diagnostics : `obj`, optional
            An instance of either

                * :class:`sympl.DiagnosticComponent`,
                * :class:`sympl.DiagnosticComponentComposite`, or
                * :class:`tasmania.DiagnosticComponentComposite`

            retrieving diagnostics at the end of each stage, once the
            sub-timestepping routine is over.
        substeps : `int`, optional
            Number of sub-steps to perform. Defaults to 0, meaning that no
            sub-stepping technique is implemented.
        fast_tendencies : `obj`, optional
            An instance of either

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            calculating the fast physical tendencies.
            Here, *fast* refers to the fact that these physical packages are
            called *before* each sub-step of any stage of the dynamical core
            to calculate the physical tendencies.
            This parameter is ignored if `substeps` argument is not positive.
        fast_diagnostics : `obj`, optional
            An instance of either

                * :class:`sympl.DiagnosticComponent`,
                * :class:`sympl.DiagnosticComponentComposite`, or
                * :class:`tasmania.DiagnosticComponentComposite`

            retrieving diagnostics at the end of each sub-step of any stage
            of the dynamical core.
            This parameter is ignored if `substeps` argument is not positive.
        moist : bool
            `True` for a moist dynamical core, `False` otherwise.
            Defaults to `False`.
        time_integration_scheme : str
            String specifying the time stepping method to implement.
            See :class:`tasmania.IsentropiPrognostic`
            for all available options. Defaults to 'forward_euler'.
        horizontal_flux_scheme : str
            String specifying the numerical horizontal flux to use.
            See :class:`tasmania.HorizontalIsentropicFlux`
            for all available options. Defaults to 'upwind'.
        time_integration_properties : dict
            Additional properties to be passed to the constructor of
            :class:`tasmania.IsentropicPrognostic` as keyword arguments.
        damp : `bool`, optional
            `True` to enable vertical damping, `False` otherwise.
            Defaults to `True`.
        damp_at_every_stage : `bool`, optional
            `True` to carry out the damping at each stage of the multi-stage
            time-integrator, `False` to carry out the damping only at the end
            of each timestep. Defaults to `True`.
        damp_type : `str`, optional
            String specifying the vertical damping scheme to implement.
            See :class:`tasmania.VerticalDamping` for all available options.
            Defaults to 'rayleigh'.
        damp_depth : `int`, optional
            Number of vertical layers in the damping region. Defaults to 15.
        damp_max : `float`, optional
            Maximum value for the damping coefficient. Defaults to 0.0002.
        smooth : `bool`, optional
            `True` to enable horizontal numerical smoothing, `False` otherwise.
            Defaults to `True`.
        smooth_at_every_stage : `bool`, optional
            `True` to apply numerical smoothing at each stage of the time-
            integrator, `False` to apply numerical smoothing only at the end
            of each timestep. Defaults to `True`.
        smooth_type: `str`, optional
            String specifying the smoothing technique to implement.
            See :class:`tasmania.HorizontalSmoothing` for all available options.
            Defaults to 'first_order'.
        smooth_coeff : `float`, optional
            Smoothing coefficient. Defaults to 0.03.
        smooth_coeff_max : `float`, optional
            Maximum value for the smoothing coefficient.
            See :class:`tasmania.HorizontalSmoothing` for further details.
            Defaults to 0.24.
        smooth_damp_depth : `int`, optional
            Number of vertical layers in the smoothing damping region. Defaults to 10.
        smooth_moist : `bool`, optional
            `True` to enable horizontal numerical smoothing on the water constituents,
            `False` otherwise. Defaults to `True`.
        smooth_moist_at_every_stage : `bool`, optional
            `True` to apply numerical smoothing on the water constituents
            at each stage of the time-integrator, `False` to apply numerical
            smoothing only at the end of each timestep. Defaults to `True`.
        smooth_moist_type: `str`, optional
            String specifying the smoothing technique to apply on the water constituents.
            See :class:`tasmania.HorizontalSmoothing` for all available options.
            Defaults to 'first-order'.
        smooth_moist_coeff : `float`, optional
            Smoothing coefficient for the water constituents. Defaults to 0.03.
        smooth_moist_coeff_max : `float`, optional
            Maximum value for the smoothing coefficient for the water constituents.
            See :class:`tasmania.HorizontalSmoothing` for further details.
            Defaults to 0.24.
        smooth_moist_damp_depth : `int`, optional
            Number of vertical layers in the smoothing damping region for the
            water constituents. Defaults to 10.
        gt_powered : `bool`, optional
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        """
        #
        # set storage shape
        #
        grid = domain.numerical_grid
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        storage_shape = (
            (nx + 1, ny + 1, nz + 1) if storage_shape is None else storage_shape
        )
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx + 1, ny + 1, nz + 1)
        )
        assert storage_shape[0] >= nx + 1, error_msg
        assert storage_shape[1] >= ny + 1, error_msg
        assert storage_shape[2] >= nz + 1, error_msg

        #
        # input parameters
        #
        self._moist = moist
        self._damp = damp
        self._damp_at_every_stage = damp_at_every_stage
        self._smooth = smooth
        self._smooth_at_every_stage = smooth_at_every_stage
        self._smooth_moist = smooth_moist
        self._smooth_moist_at_every_stage = smooth_moist_at_every_stage
        self._backend = backend
        self._dtype = dtype
        self._default_origin = default_origin
        self._storage_shape = storage_shape
        self._managed_memory = managed_memory

        #
        # parent constructor
        #
        super().__init__(
            domain,
            "numerical",
            intermediate_tendencies,
            intermediate_diagnostics,
            substeps,
            fast_tendencies,
            fast_diagnostics,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
        )
        hb = self.horizontal_boundary

        #
        # prognostic
        #
        kwargs = time_integration_properties or {}
        self._prognostic = IsentropicPrognostic.factory(
            time_integration_scheme,
            horizontal_flux_scheme,
            self.grid,
            self.horizontal_boundary,
            moist,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            storage_shape=storage_shape,
            managed_memory=managed_memory,
            **kwargs
        )

        #
        # vertical damping
        #
        if damp:
            self._damper = VerticalDamping.factory(
                damp_type,
                self.grid,
                damp_depth,
                damp_max,
                gt_powered=gt_powered,
                backend=backend,
                backend_opts=backend_opts,
                build_info=build_info,
                dtype=dtype,
                exec_info=exec_info,
                default_origin=default_origin,
                rebuild=rebuild,
                storage_shape=storage_shape,
            )

        #
        # numerical smoothing
        #
        if smooth:
            self._smoother = HorizontalSmoothing.factory(
                smooth_type,
                storage_shape,
                smooth_coeff,
                smooth_coeff_max,
                smooth_damp_depth,
                hb.nb,
                gt_powered=gt_powered,
                backend=backend,
                backend_opts=backend_opts,
                build_info=build_info,
                dtype=dtype,
                exec_info=exec_info,
                default_origin=default_origin,
                rebuild=rebuild,
                managed_memory=managed_memory,
            )
            if moist and smooth_moist:
                self._smoother_moist = HorizontalSmoothing.factory(
                    smooth_moist_type,
                    storage_shape,
                    smooth_moist_coeff,
                    smooth_moist_coeff_max,
                    smooth_moist_damp_depth,
                    hb.nb,
                    gt_powered=gt_powered,
                    backend=backend,
                    backend_opts=backend_opts,
                    build_info=build_info,
                    dtype=dtype,
                    exec_info=exec_info,
                    default_origin=default_origin,
                    rebuild=rebuild,
                    managed_memory=managed_memory,
                )

        #
        # diagnostics
        #
        self._velocity_components = HorizontalVelocity(
            self.grid,
            staggering=True,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            rebuild=rebuild,
        )
        if moist:
            self._water_constituent = WaterConstituent(
                self.grid,
                clipping=True,
                gt_powered=gt_powered,
                backend=backend,
                backend_opts=backend_opts,
                build_info=build_info,
                dtype=dtype,
                exec_info=exec_info,
                rebuild=rebuild,
            )

        #
        # the method implementing each stage
        #
        self._array_call = self._array_call_dry if not moist else self._array_call_moist

        #
        # temporary and output arrays
        #
        def allocate():
            return zeros(
                storage_shape,
                gt_powered=gt_powered,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )

        if moist:
            self._sqv_now = allocate()
            self._sqc_now = allocate()
            self._sqr_now = allocate()
            self._sqv_int = allocate()
            self._sqc_int = allocate()
            self._sqr_int = allocate()
            self._qv_new = allocate()
            self._qc_new = allocate()
            self._qr_new = allocate()

        if damp:
            self._s_ref = allocate()
            self._s_damped = allocate()
            self._su_ref = allocate()
            self._su_damped = allocate()
            self._sv_ref = allocate()
            self._sv_damped = allocate()
            # if moist:
            #     self._qv_ref = allocate()
            #     self._qv_damped = allocate()
            #     self._qc_ref = allocate()
            #     self._qc_damped = allocate()
            #     self._qr_ref = allocate()
            #     self._qr_damped = allocate()

        if smooth:
            self._s_smoothed = allocate()
            self._su_smoothed = allocate()
            self._sv_smoothed = allocate()

        if smooth_moist:
            self._qv_smoothed = allocate()
            self._qc_smoothed = allocate()
            self._qr_smoothed = allocate()

        self._u_out = allocate()
        self._v_out = allocate()

    @property
    def _input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "montgomery_potential": {"dims": dims, "units": "m^2 s^-2"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "x_velocity_at_u_locations": {"dims": dims_stg_x, "units": "m s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_velocity_at_v_locations": {"dims": dims_stg_y, "units": "m s^-1"},
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def _substep_input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        ftends, fdiags = self._fast_tends, self._fast_diags

        return_dict = {}

        if (
            ftends is not None
            and "air_isentropic_density" in ftends.input_properties
            or ftends is not None
            and "air_isentropic_density" in ftends.tendency_properties
            or fdiags is not None
            and "air_isentropic_density" in fdiags.input_properties
        ):
            return_dict["air_isentropic_density"] = {
                "dims": dims,
                "units": "kg m^-2 K^-1",
            }

        if (
            ftends is not None
            and "x_momentum_isentropic" in ftends.input_properties
            or ftends is not None
            and "x_momentum_isentropic" in ftends.tendency_properties
            or fdiags is not None
            and "x_momentum_isentropic" in fdiags.input_properties
        ):
            return_dict["x_momentum_isentropic"] = {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            }

        if (
            ftends is not None
            and "y_momentum_isentropic" in ftends.input_properties
            or ftends is not None
            and "y_momentum_isentropic" in ftends.tendency_properties
            or fdiags is not None
            and "y_momentum_isentropic" in fdiags.input_properties
        ):
            return_dict["y_momentum_isentropic"] = {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            }

        if self._moist:
            if (
                ftends is not None
                and mfwv in ftends.input_properties
                or ftends is not None
                and mfwv in ftends.tendency_properties
                or fdiags is not None
                and mfwv in fdiags.input_properties
            ):
                return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}

            if (
                ftends is not None
                and mfcw in ftends.input_properties
                or ftends is not None
                and mfcw in ftends.tendency_properties
                or fdiags is not None
                and mfcw in fdiags.input_properties
            ):
                return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}

            if (
                ftends is not None
                and mfpw in ftends.input_properties
                or ftends is not None
                and mfpw in ftends.tendency_properties
                or fdiags is not None
                and mfpw in fdiags.input_properties
            ):
                return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

            if ftends is not None and "precipitation" in ftends.diagnostic_properties:
                dims2d = (self.grid.x.dims[0], self.grid.y.dims[0])
                return_dict.update(
                    {
                        "precipitation": {"dims": dims2d, "units": "mm hr^-1"},
                        "accumulated_precipitation": {"dims": dims2d, "units": "mm"},
                    }
                )

        return return_dict

    @property
    def _tendency_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1 s^-1"}

        return return_dict

    @property
    def _substep_tendency_properties(self) -> taz_types.properties_dict_t:
        return self._tendency_properties

    @property
    def _output_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])
        g_shape = (g.nx, g.ny, g.nz)
        g_shape_stg_x = (g.nx + 1, g.ny, g.nz)
        g_shape_stg_y = (g.nx, g.ny + 1, g.nz)

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1",
                "grid_shape": g_shape,
            },
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
                "grid_shape": g_shape,
            },
            "x_velocity_at_u_locations": {
                "dims": dims_stg_x,
                "units": "m s^-1",
                "grid_shape": g_shape_stg_x,
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
                "grid_shape": g_shape,
            },
            "y_velocity_at_v_locations": {
                "dims": dims_stg_y,
                "units": "m s^-1",
                "grid_shape": g_shape_stg_y,
            },
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1", "grid_shape": g_shape}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1", "grid_shape": g_shape}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1", "grid_shape": g_shape}

        return return_dict

    @property
    def _substep_output_properties(self) -> taz_types.properties_dict_t:
        if not hasattr(self, "__substep_output_properties"):
            dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
            g_shape = (self.grid.nx, self.grid.ny, self.grid.nz)

            self.__substep_output_properties = {}

            if "air_isentropic_density" in self._substep_input_properties:
                self.__substep_output_properties["air_isentropic_density"] = {
                    "dims": dims,
                    "units": "kg m^-2 K^-1",
                    "grid_shape": g_shape,
                }

            if "x_momentum_isentropic" in self._substep_input_properties:
                self.__substep_output_properties["x_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                    "grid_shape": g_shape,
                }

            if "y_momentum_isentropic" in self._substep_input_properties:
                self.__substep_output_properties["y_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                    "grid_shape": g_shape,
                }

            if self._moist:
                if mfwv in self._substep_input_properties:
                    self.__substep_output_properties[mfwv] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if mfcw in self._substep_input_properties:
                    self.__substep_output_properties[mfcw] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if mfpw in self._substep_input_properties:
                    self.__substep_output_properties[mfpw] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if "precipitation" in self._substep_input_properties:
                    dims2d = (self.grid.x.dims[0], self.grid.y.dims[0], 1)
                    g_shape_2d = (self.grid.nx, self.grid.ny, 1)
                    self.__substep_output_properties["accumulated_precipitation"] = {
                        "dims": dims2d,
                        "units": "mm",
                        "grid_shape": g_shape_2d,
                    }

        return self.__substep_output_properties

    @property
    def stages(self) -> int:
        return self._prognostic.stages

    @property
    def substep_fractions(self) -> Union[float, Tuple[float, ...]]:
        return self._prognostic.substep_fractions

    def _allocate_output_state(self) -> taz_types.gtstorage_dict_t:
        """ Allocate memory only for the prognostic fields. """
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz
        gt_powered = self._gt_powered
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        storage_shape = self._storage_shape
        managed_memory = self._managed_memory

        out_state = {}

        names = [
            "air_isentropic_density",
            "x_velocity_at_u_locations",
            "x_momentum_isentropic",
            "y_velocity_at_v_locations",
            "y_momentum_isentropic",
        ]
        if self._moist:
            names.append(mfwv)
            names.append(mfcw)
            names.append(mfpw)

        for name in names:
            dims = self.output_properties[name]["dims"]
            units = self.output_properties[name]["units"]

            grid_shape = (
                nx + 1 if "at_u_locations" in dims[0] else nx,
                ny + 1 if "at_v_locations" in dims[1] else ny,
                nz + 1 if "on_interface_levels" in dims[2] else nz,
            )

            out_state[name] = get_dataarray_3d(
                zeros(
                    storage_shape,
                    gt_powered=gt_powered,
                    backend=backend,
                    dtype=dtype,
                    default_origin=default_origin,
                    managed_memory=managed_memory,
                ),
                g,
                units,
                name=name,
                grid_shape=grid_shape,
                set_coordinates=False,
            )

        return out_state

    def array_call(
        self,
        stage: int,
        raw_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
        return self._array_call(stage, raw_state, raw_tendencies, timestep)

    def _array_call_dry(
        self,
        stage: int,
        raw_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
        """ Perform a stage of the dry dynamical core. """
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref[...] = (
                    ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
                )
                self._su_ref[...] = (
                    ref_state["x_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._sv_ref[...] = (
                    ref_state["y_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the horizontal "
                    "boundary conditions, but needed by the wave absorber."
                )

            # save the current solution
            self._s_now = raw_state["air_isentropic_density"]
            self._su_now = raw_state["x_momentum_isentropic"]
            self._sv_now = raw_state["y_momentum_isentropic"]

        # perform the prognostic step
        raw_state_new = self._prognostic.stage_call(
            stage, timestep, raw_state, raw_tendencies
        )

        # apply the lateral boundary conditions
        hb.dmn_enforce_raw(raw_state_new, out_properties)

        # extract the stepped prognostic model variables
        s_new = raw_state_new["air_isentropic_density"]
        su_new = raw_state_new["x_momentum_isentropic"]
        sv_new = raw_state_new["y_momentum_isentropic"]

        damped = False
        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            damped = True

            # apply vertical damping
            self._damper(timestep, self._s_now, s_new, self._s_ref, self._s_damped)
            self._damper(timestep, self._su_now, su_new, self._su_ref, self._su_damped)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, self._sv_damped)

        # properly set pointers to current solution
        s_new = self._s_damped if damped else s_new
        su_new = self._su_damped if damped else su_new
        sv_new = self._sv_damped if damped else sv_new

        smoothed = False
        if self._smooth and (self._smooth_at_every_stage or stage == self.stages - 1):
            smoothed = True

            # apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                "air_isentropic_density": self._s_smoothed,
                "x_momentum_isentropic": self._su_smoothed,
                "y_momentum_isentropic": self._sv_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new

        # diagnose the velocity components
        self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out, self._u_out, self._v_out
        )
        hb.dmn_set_outermost_layers_x(
            self._u_out,
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=raw_state_new["time"],
        )
        hb.dmn_set_outermost_layers_y(
            self._v_out,
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=raw_state_new["time"],
        )

        # instantiate the output state
        raw_state_out = {
            "time": raw_state_new["time"],
            "air_isentropic_density": s_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": self._u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": self._v_out,
        }

        return raw_state_out

    def _array_call_moist(
        self,
        stage: int,
        raw_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
        """	Perform a stage of the moist dynamical core. """
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref[...] = (
                    ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
                )
                self._su_ref[...] = (
                    ref_state["x_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )
                self._sv_ref[...] = (
                    ref_state["y_momentum_isentropic"]
                    .to_units("kg m^-1 K^-1 s^-1")
                    .values
                )

                # self._qv_ref[...] = (
                #     ref_state[mfwv].to_units("g g^-1").values
                # )
                # self._qc_ref[...] = (
                #     ref_state[mfcw].to_units("g g^-1").values
                # )
                # self._qr_ref[...] = (
                #     ref_state[mfpw].to_units("g g^-1").values
                # )
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the horizontal "
                    "boundary conditions, but needed by the wave absorber."
                )

            # save the current solution
            self._s_now = raw_state["air_isentropic_density"]
            self._su_now = raw_state["x_momentum_isentropic"]
            self._sv_now = raw_state["y_momentum_isentropic"]
            # self._qv_now = raw_state[mfwv]
            # self._qc_now = raw_state[mfcw]
            # self._qr_now = raw_state[mfpw]

        s_now = raw_state["air_isentropic_density"]
        qv_now = raw_state[mfwv]
        qc_now = raw_state[mfcw]
        qr_now = raw_state[mfpw]

        # diagnose the isentropic density of all water constituents
        sqv = self._sqv_now if stage == 0 else self._sqv_int
        sqc = self._sqc_now if stage == 0 else self._sqc_int
        sqr = self._sqr_now if stage == 0 else self._sqr_int
        self._water_constituent.get_density_of_water_constituent(s_now, qv_now, sqv)
        self._water_constituent.get_density_of_water_constituent(s_now, qc_now, sqc)
        self._water_constituent.get_density_of_water_constituent(s_now, qr_now, sqr)
        raw_state["isentropic_density_of_water_vapor"] = sqv
        raw_state["isentropic_density_of_cloud_liquid_water"] = sqc
        raw_state["isentropic_density_of_precipitation_water"] = sqr

        # perform the prognostic step
        raw_state_new = self._prognostic.stage_call(
            stage, timestep, raw_state, raw_tendencies
        )

        # extract the stepped prognostic model variables
        s_new = raw_state_new["air_isentropic_density"]
        su_new = raw_state_new["x_momentum_isentropic"]
        sv_new = raw_state_new["y_momentum_isentropic"]
        sqv_new = raw_state_new["isentropic_density_of_water_vapor"]
        sqc_new = raw_state_new["isentropic_density_of_cloud_liquid_water"]
        sqr_new = raw_state_new["isentropic_density_of_precipitation_water"]

        # diagnose the mass fraction of all water constituents
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqv_new, self._qv_new
        )
        raw_state_new[mfwv] = self._qv_new
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqc_new, self._qc_new
        )
        raw_state_new[mfcw] = self._qc_new
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqr_new, self._qr_new
        )
        raw_state_new[mfpw] = self._qr_new

        # apply the lateral boundary conditions
        hb.dmn_enforce_raw(raw_state_new, out_properties)

        damped = False
        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            damped = True

            # apply vertical damping
            self._damper(timestep, self._s_now, s_new, self._s_ref, self._s_damped)
            self._damper(timestep, self._su_now, su_new, self._su_ref, self._su_damped)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, self._sv_damped)
            # self._damper(timestep, self._qv_now, self._qv_new, self._qv_ref, self._qv_damped)
            # self._damper(timestep, self._qc_now, self._qc_new, self._qc_ref, self._qc_damped)
            # self._damper(timestep, self._qr_now, self._qr_new, self._qr_ref, self._qr_damped)

        # properly set pointers to current solution
        s_new = self._s_damped if damped else s_new
        su_new = self._su_damped if damped else su_new
        sv_new = self._sv_damped if damped else sv_new
        qv_new = self._qv_new  # self._qv_damped if damped else self._qv_new
        qc_new = self._qc_new  # self._qc_damped if damped else self._qc_new
        qr_new = self._qr_new  # self._qr_damped if damped else self._qr_new

        smoothed = False
        if self._smooth and (self._smooth_at_every_stage or stage == self.stages - 1):
            smoothed = True

            # apply horizontal smoothing
            self._smoother(s_new, self._s_smoothed)
            self._smoother(su_new, self._su_smoothed)
            self._smoother(sv_new, self._sv_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                "air_isentropic_density": self._s_smoothed,
                "x_momentum_isentropic": self._su_smoothed,
                "y_momentum_isentropic": self._sv_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        s_out = self._s_smoothed if smoothed else s_new
        su_out = self._su_smoothed if smoothed else su_new
        sv_out = self._sv_smoothed if smoothed else sv_new

        smoothed_moist = False
        if self._smooth_moist and (
            self._smooth_moist_at_every_stage or stage == self.stages - 1
        ):
            smoothed_moist = True

            # apply horizontal smoothing
            self._smoother_moist(qv_new, self._qv_smoothed)
            self._smoother_moist(qc_new, self._qc_smoothed)
            self._smoother_moist(qr_new, self._qr_smoothed)

            # apply horizontal boundary conditions
            raw_state_smoothed = {
                "time": raw_state_new["time"],
                mfwv: self._qv_smoothed,
                mfcw: self._qc_smoothed,
                mfpw: self._qr_smoothed,
            }
            hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

        # properly set pointers to output solution
        qv_out = self._qv_smoothed if smoothed_moist else qv_new
        qc_out = self._qc_smoothed if smoothed_moist else qc_new
        qr_out = self._qr_smoothed if smoothed_moist else qr_new

        # diagnose the velocity components
        self._velocity_components.get_velocity_components(
            s_out, su_out, sv_out, self._u_out, self._v_out
        )
        hb.dmn_set_outermost_layers_x(
            self._u_out,
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=raw_state_new["time"],
        )
        hb.dmn_set_outermost_layers_y(
            self._v_out,
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=raw_state_new["time"],
        )

        # instantiate the output state
        raw_state_out = {
            "time": raw_state_new["time"],
            "air_isentropic_density": s_out,
            mfwv: qv_out,
            mfcw: qc_out,
            mfpw: qr_out,
            "x_momentum_isentropic": su_out,
            "x_velocity_at_u_locations": self._u_out,
            "y_momentum_isentropic": sv_out,
            "y_velocity_at_v_locations": self._v_out,
        }

        return raw_state_out

    def substep_array_call(
        self,
        stage: int,
        substep: int,
        raw_state: taz_types.array_dict_t,
        raw_stage_state: taz_types.array_dict_t,
        raw_tmp_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
        raise NotImplementedError()
