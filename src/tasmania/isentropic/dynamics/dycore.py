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
from typing import Optional, Sequence, TYPE_CHECKING, Tuple, Union

from sympl._core.time import FakeTimer as Timer

# from sympl._core.time import Timer

from tasmania.python.dwarfs.diagnostics import (
    HorizontalVelocity,
    WaterConstituent,
)
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.utils import typingx as ty

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import (
        DiagnosticComponent,
        TendencyComponent,
        TimeDelta,
    )


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
        fast_tendency_component: Optional["TendencyComponent"] = None,
        fast_diagnostic_component: Optional[
            Union["DiagnosticComponent", "TendencyComponent"]
        ] = None,
        substeps: int = 0,
        superfast_tendency_component: Optional["TendencyComponent"] = None,
        superfast_diagnostic_component: Optional["DiagnosticComponent"] = None,
        moist: bool = False,
        time_integration_scheme: str = "forward_euler_si",
        horizontal_flux_scheme: str = "upwind",
        time_integration_properties: Optional[ty.options_dict_t] = None,
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
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        fast_tendency_component : `obj`, optional
            An instance of either

            * :class:`~sympl.TendencyComponent`,
            * :class:`~sympl.TendencyComponentComposite`,
            * :class:`~sympl.ImplicitTendencyComponent`,
            * :class:`~sympl.ImplicitTendencyComponentComposite`, or
            * :class:`~tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each stage on the latest
            provisional state.
        fast_diagnostic_component : `obj`, optional
            An instance of either

            * :class:`sympl.TendencyComponent`,
            * :class:`sympl.TendencyComponentComposite`,
            * :class:`sympl.ImplicitTendencyComponent`,
            * :class:`sympl.ImplicitTendencyComponentComposite`,
            * :class:`tasmania.ConcurrentCoupling`,
            * :class:`sympl.DiagnosticComponent`,
            * :class:`sympl.DiagnosticComponentComposite`, or
            * :class:`tasmania.DiagnosticComponentComposite`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the end of each stage on the latest
            provisional state, once the substepping routine is over.
        substeps : `int`, optional
            Number of substeps to perform. Defaults to 0, meaning that no
            form of substepping is carried out.
        superfast_tendency_component : `obj`, optional
            An instance of either

            * :class:`sympl.TendencyComponent`,
            * :class:`sympl.TendencyComponentComposite`,
            * :class:`sympl.ImplicitTendencyComponent`,
            * :class:`sympl.ImplicitTendencyComponentComposite`, or
            * :class:`tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each substep on the
            latest provisional state. This parameter is ignored if ``substeps``
            is not positive.
        superfast_diagnostic_component : `obj`, optional
            An instance of either

            * :class:`sympl.DiagnosticComponent`,
            * :class:`sympl.DiagnosticComponentComposite`, or
            * :class:`tasmania.DiagnosticComponentComposite`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the end of each substep on the latest
            provisional state.
        moist : bool
            ``True`` for a moist dynamical core, ``False`` otherwise.
            Defaults to ``False``.
        time_integration_scheme : str
            String specifying the time stepping method to implement.
            See :class:`~tasmania.IsentropiPrognostic`
            for all available options. Defaults to "forward_euler".
        horizontal_flux_scheme : str
            String specifying the numerical horizontal flux to use.
            See :class:`~tasmania.IsentropicHorizontalFlux`
            for all available options. Defaults to "upwind".
        time_integration_properties : dict
            Additional properties to be passed to the constructor of
            :class:`~tasmania.IsentropicPrognostic` as keyword arguments.
        damp : `bool`, optional
            ``True`` to enable vertical damping, ``False`` otherwise.
            Defaults to ``True``.
        damp_at_every_stage : `bool`, optional
            ``True`` to carry out the damping at each stage of the multi-stage
            time-integrator, ``False`` to carry out the damping only at the end
            of each timestep. Defaults to ``True``.
        damp_type : `str`, optional
            String specifying the vertical damping scheme to implement.
            See :class:`~tasmania.VerticalDamping` for all available options.
            Defaults to "rayleigh".
        damp_depth : `int`, optional
            Number of vertical layers in the damping region. Defaults to 15.
        damp_max : `float`, optional
            Maximum value for the damping coefficient. Defaults to 0.0002.
        smooth : `bool`, optional
            ``True`` to enable horizontal numerical smoothing,
            ``False`` otherwise.
            Defaults to ``True``.
        smooth_at_every_stage : `bool`, optional
            ``True`` to apply numerical smoothing at each stage of the time-
            integrator, ``False`` to apply numerical smoothing only at the end
            of each timestep. Defaults to ``True``.
        smooth_type: `str`, optional
            String specifying the smoothing technique to implement.
            See :class:`~tasmania.HorizontalSmoothing` for all available
            options. Defaults to "first_order".
        smooth_coeff : `float`, optional
            Smoothing coefficient. Defaults to 0.03.
        smooth_coeff_max : `float`, optional
            Maximum value for the smoothing coefficient.
            See :class:`~tasmania.HorizontalSmoothing` for further details.
            Defaults to 0.24.
        smooth_damp_depth : `int`, optional
            Number of vertical layers in the smoothing damping region.
            Defaults to 10.
        smooth_moist : `bool`, optional
            ``True`` to enable horizontal numerical smoothing on the water
            constituents, ``False`` otherwise. Defaults to ``True``.
        smooth_moist_at_every_stage : `bool`, optional
            ``True`` to apply numerical smoothing on the water constituents
            at each stage of the time-integrator, ``False`` to apply numerical
            smoothing only at the end of each timestep. Defaults to ``True``.
        smooth_moist_type: `str`, optional
            String specifying the smoothing technique to apply on the water
            constituents. See :class:`~tasmania.HorizontalSmoothing` for all
            available options. Defaults to "first-order".
        smooth_moist_coeff : `float`, optional
            Smoothing coefficient for the water constituents. Defaults to 0.03.
        smooth_moist_coeff_max : `float`, optional
            Maximum value for the smoothing coefficient for the water
            constituents. See :class:`tasmania.HorizontalSmoothing` for further
            details. Defaults to 0.24.
        smooth_moist_damp_depth : `int`, optional
            Number of vertical layers in the smoothing damping region for the
            water constituents. Defaults to 10.
        enable_checks : `bool`, optional
            TODO
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
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

        #
        # parent constructor
        #
        g = domain.numerical_grid
        super().__init__(
            domain,
            fast_tendency_component,
            fast_diagnostic_component,
            substeps,
            superfast_tendency_component,
            superfast_diagnostic_component,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=self.get_storage_shape(storage_shape, (g.nx + 1, g.ny + 1, g.nz + 1)),
            storage_options=storage_options,
        )

        #
        # prognostic
        #
        kwargs = time_integration_properties or {}
        self._prognostic = IsentropicPrognostic.factory(
            time_integration_scheme,
            horizontal_flux_scheme,
            domain,
            moist,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_shape=self.storage_shape,
            storage_options=self.storage_options,
            **kwargs,
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
                backend=self.backend,
                backend_options=self.backend_options,
                storage_shape=self.storage_shape,
                storage_options=self.storage_options,
            )

        #
        # diagnostics
        #
        self._velocity_components = HorizontalVelocity(
            self.grid,
            staggering=True,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_options=self.storage_options,
        )
        if moist:
            self._water_constituent = WaterConstituent(
                self.grid,
                clipping=True,
                backend=self.backend,
                backend_options=self.backend_options,
                storage_options=self.storage_options,
            )

        #
        # temporary and output arrays
        #
        self._s_now = None
        self._su_now = None
        self._sv_now = None

        if moist:
            self._sqv_now = self.zeros(shape=storage_shape)
            self._sqc_now = self.zeros(shape=storage_shape)
            self._sqr_now = self.zeros(shape=storage_shape)
            self._sqv_int = self.zeros(shape=storage_shape)
            self._sqc_int = self.zeros(shape=storage_shape)
            self._sqr_int = self.zeros(shape=storage_shape)
            self._sqv_new = self.zeros(shape=storage_shape)
            self._sqc_new = self.zeros(shape=storage_shape)
            self._sqr_new = self.zeros(shape=storage_shape)

        if damp:
            self._s_ref = self.zeros(shape=storage_shape)
            self._su_ref = self.zeros(shape=storage_shape)
            self._sv_ref = self.zeros(shape=storage_shape)

    @property
    def stage_input_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stg_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "montgomery_potential": {"dims": dims, "units": "m^2 s^-2"},
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
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def substep_input_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        ftends, fdiags = self._superfast_tc, self._superfast_dc

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
                        "accumulated_precipitation": {
                            "dims": dims2d,
                            "units": "mm",
                        },
                    }
                )

        return return_dict

    @property
    def stage_tendency_properties(self) -> "PropertyDict":
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {
                "dims": dims,
                "units": "kg m^-2 K^-1 s^-1",
            },
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1 s^-1"}

        return return_dict

    @property
    def substep_tendency_properties(self) -> "PropertyDict":
        return self.stage_tendency_properties

    @property
    def stage_output_properties(self) -> "PropertyDict":
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
            return_dict[mfwv] = {
                "dims": dims,
                "units": "g g^-1",
                "grid_shape": g_shape,
            }
            return_dict[mfcw] = {
                "dims": dims,
                "units": "g g^-1",
                "grid_shape": g_shape,
            }
            return_dict[mfpw] = {
                "dims": dims,
                "units": "g g^-1",
                "grid_shape": g_shape,
            }

        return return_dict

    @property
    def substep_output_properties(self) -> "PropertyDict":
        if not hasattr(self, "__substep_output_properties"):
            dims = (
                self.grid.x.dims[0],
                self.grid.y.dims[0],
                self.grid.z.dims[0],
            )
            g_shape = (self.grid.nx, self.grid.ny, self.grid.nz)

            self.__substep_output_properties = {}

            if "air_isentropic_density" in self.substep_input_properties:
                self.__substep_output_properties["air_isentropic_density"] = {
                    "dims": dims,
                    "units": "kg m^-2 K^-1",
                    "grid_shape": g_shape,
                }

            if "x_momentum_isentropic" in self.substep_input_properties:
                self.__substep_output_properties["x_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                    "grid_shape": g_shape,
                }

            if "y_momentum_isentropic" in self.substep_input_properties:
                self.__substep_output_properties["y_momentum_isentropic"] = {
                    "dims": dims,
                    "units": "kg m^-1 K^-1 s^-1",
                    "grid_shape": g_shape,
                }

            if self._moist:
                if mfwv in self.substep_input_properties:
                    self.__substep_output_properties[mfwv] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if mfcw in self.substep_input_properties:
                    self.__substep_output_properties[mfcw] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if mfpw in self.substep_input_properties:
                    self.__substep_output_properties[mfpw] = {
                        "dims": dims,
                        "units": "g g^-1",
                        "grid_shape": g_shape,
                    }

                if "precipitation" in self.substep_input_properties:
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

    def stage_array_call(
        self,
        stage: int,
        state: "NDArrayLikeDict",
        tendencies: "NDArrayLikeDict",
        timestep: "TimeDelta",
        out_state: "NDArrayLikeDict",
    ) -> None:
        if self._moist:
            self.stage_array_call_moist(stage, state, tendencies, timestep, out_state)
        else:
            self.stage_array_call_dry(stage, state, tendencies, timestep, out_state)

    def stage_array_call_dry(
        self,
        stage: int,
        state: "NDArrayLikeDict",
        tendencies: "NDArrayLikeDict",
        timestep: "TimeDelta",
        out_state: "NDArrayLikeDict",
    ) -> None:
        """Integrate the state over a stage of the dry dynamical core."""
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            Timer.start(label="set_reference_state")
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref[...] = ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
                self._su_ref[...] = (
                    ref_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
                )
                self._sv_ref[...] = (
                    ref_state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
                )
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the "
                    "horizontal boundary conditions, but needed by the "
                    "wave absorber."
                )
            Timer.stop()

            # save the current solution
            self._s_now = state["air_isentropic_density"]
            self._su_now = state["x_momentum_isentropic"]
            self._sv_now = state["y_momentum_isentropic"]

        # perform the prognostic step
        Timer.start(label="prognostic")
        self._prognostic.stage_call(stage, timestep, state, tendencies, out_state)
        Timer.stop()

        # apply the lateral boundary conditions
        Timer.start(label="boundary_conditions")
        hb.enforce_raw(out_state, out_properties)
        Timer.stop()

        # extract the stepped prognostic model variables
        s_new = out_state["air_isentropic_density"]
        su_new = out_state["x_momentum_isentropic"]
        sv_new = out_state["y_momentum_isentropic"]

        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            Timer.start(label="vertical_damping")
            # apply vertical damping
            self._damper(timestep, self._s_now, s_new, self._s_ref, s_new)
            self._damper(timestep, self._su_now, su_new, self._su_ref, su_new)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, sv_new)
            Timer.stop()

        # diagnose the velocity components
        self._velocity_components.get_velocity_components(
            s_new,
            su_new,
            sv_new,
            out_state["x_velocity_at_u_locations"],
            out_state["y_velocity_at_v_locations"],
        )
        hb.set_outermost_layers_x(
            out_state["x_velocity_at_u_locations"],
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=out_state["time"],
        )
        hb.set_outermost_layers_y(
            out_state["y_velocity_at_v_locations"],
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=out_state["time"],
        )

    def stage_array_call_moist(
        self,
        stage: int,
        state: "NDArrayLikeDict",
        tendencies: "NDArrayLikeDict",
        timestep: "TimeDelta",
        out_state: "NDArrayLikeDict",
    ) -> None:
        """Integrate the state over a stage of the moist dynamical core."""
        # shortcuts
        hb = self.horizontal_boundary
        out_properties = self.output_properties

        if self._damp and stage == 0:
            Timer.start(label="set_reference_state")
            # set the reference state
            try:
                ref_state = hb.reference_state
                self._s_ref[...] = ref_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
                self._su_ref[...] = (
                    ref_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
                )
                self._sv_ref[...] = (
                    ref_state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
                )
            except KeyError:
                raise RuntimeError(
                    "Reference state not set in the object handling the "
                    "horizontal boundary conditions, but needed by the "
                    "wave absorber."
                )
            Timer.stop()

            # save the current solution
            self._s_now = state["air_isentropic_density"]
            self._su_now = state["x_momentum_isentropic"]
            self._sv_now = state["y_momentum_isentropic"]

        # diagnose the isentropic density of all water constituents
        Timer.start(label="diagnose_sq")
        sqv = self._sqv_now if stage == 0 else self._sqv_int
        sqc = self._sqc_now if stage == 0 else self._sqc_int
        sqr = self._sqr_now if stage == 0 else self._sqr_int
        self._water_constituent.get_density_of_water_constituent(
            state["air_isentropic_density"], state[mfwv], sqv
        )
        self._water_constituent.get_density_of_water_constituent(
            state["air_isentropic_density"], state[mfcw], sqc
        )
        self._water_constituent.get_density_of_water_constituent(
            state["air_isentropic_density"], state[mfpw], sqr
        )
        state["isentropic_density_of_water_vapor"] = sqv
        state["isentropic_density_of_cloud_liquid_water"] = sqc
        state["isentropic_density_of_precipitation_water"] = sqr
        out_state["isentropic_density_of_water_vapor"] = self._sqv_new
        out_state["isentropic_density_of_cloud_liquid_water"] = self._sqc_new
        out_state["isentropic_density_of_precipitation_water"] = self._sqr_new
        Timer.stop()

        # perform the prognostic step
        Timer.start(label="prognostic")
        self._prognostic.stage_call(stage, timestep, state, tendencies, out_state)
        Timer.stop()

        # extract the stepped prognostic model variables
        s_new = out_state["air_isentropic_density"]
        su_new = out_state["x_momentum_isentropic"]
        sv_new = out_state["y_momentum_isentropic"]
        sqv_new = out_state.pop("isentropic_density_of_water_vapor")
        sqc_new = out_state.pop("isentropic_density_of_cloud_liquid_water")
        sqr_new = out_state.pop("isentropic_density_of_precipitation_water")

        # diagnose the mass fraction of all water constituents
        Timer.start(label="diagnose_q")
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqv_new, out_state[mfwv]
        )
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqc_new, out_state[mfcw]
        )
        self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
            s_new, sqr_new, out_state[mfpw]
        )
        Timer.stop()

        # apply the lateral boundary conditions
        Timer.start(label="boundary_conditions")
        hb.enforce_raw(out_state, out_properties)
        Timer.stop()

        if self._damp and (self._damp_at_every_stage or stage == self.stages - 1):
            # apply vertical damping
            Timer.start(label="vertical_damping")
            self._damper(timestep, self._s_now, s_new, self._s_ref, s_new)
            self._damper(timestep, self._su_now, su_new, self._su_ref, su_new)
            self._damper(timestep, self._sv_now, sv_new, self._sv_ref, sv_new)
            Timer.stop()

        # diagnose the velocity components
        Timer.start(label="velocity")
        self._velocity_components.get_velocity_components(
            s_new,
            su_new,
            sv_new,
            out_state["x_velocity_at_u_locations"],
            out_state["y_velocity_at_v_locations"],
        )
        hb.set_outermost_layers_x(
            out_state["x_velocity_at_u_locations"],
            field_name="x_velocity_at_u_locations",
            field_units=out_properties["x_velocity_at_u_locations"]["units"],
            time=out_state["time"],
        )
        hb.set_outermost_layers_y(
            out_state["y_velocity_at_v_locations"],
            field_name="y_velocity_at_v_locations",
            field_units=out_properties["y_velocity_at_v_locations"]["units"],
            time=out_state["time"],
        )
        Timer.stop()

    def substep_array_call(
        self,
        stage: int,
        substep: int,
        raw_state: "NDArrayLikeDict",
        raw_stage_state: "NDArrayLikeDict",
        raw_tmp_state: "NDArrayLikeDict",
        raw_tendencies: "NDArrayLikeDict",
        timestep: "TimeDelta",
    ) -> "NDArrayLikeDict":
        raise NotImplementedError()
