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
from sympl import DataArray
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics as Core,
)
from tasmania.python.utils import typing

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class IsentropicDiagnostics(DiagnosticComponent):
    """
    With the help of the isentropic density, this class diagnoses

        * the pressure,
        * the Exner function,
        * the Montgomery potential and
        * the height of the interface levels.

    Optionally,

        * the air density and
        * the air temperature

    are calculated as well.
    """

    # default values for the physical constants used in the class
    default_physical_constants = {
        "air_pressure_at_sea_level": DataArray(1e5, attrs={"units": "Pa"}),
        "gas_constant_of_dry_air": DataArray(
            287.05, attrs={"units": "J K^-1 kg^-1"}
        ),
        "gravitational_acceleration": DataArray(
            9.80665, attrs={"units": "m s^-2"}
        ),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        moist: bool,
        pt: DataArray,
        physical_constants: Optional[Mapping[str, DataArray]] = None,
        *,
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
        grid_type : str
            The type of grid over which instantiating the class.
            Either "physical" or "numerical".
        moist : bool
            ``True`` if water species are included in the model,
            ``False`` otherwise.
        pt : sympl.DataArray
            One-item :class:`sympl.DataArray` representing the air pressure
            at the top edge of the domain.
        physical_constants : `dict[str, sympl.DataArray]`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants.
            The constants might be:

                * 'air_pressure_at_sea_level', in units compatible with [Pa];
                * 'gas_constant_of_dry_air', in units compatible with
                    [J K^-1 kg^-1];
                * 'gravitational_acceleration', in units compatible with
                    [m s^-2];
                * 'specific_heat_of_dry_air_at_constant_pressure', in units
                    compatible with [J K^-1 kg^-1].

            Please refer to
            :func:`tasmania.utils.data_utils.get_physical_constants` and
            :obj:`tasmania.physics.isentropic.IsentropicDiagnostics._d_physical_constants`
            for the default values.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        # store input parameters needed at run-time
        self._moist = moist
        self._pt = pt.to_units("Pa").data.item()

        # initialize the parent class
        super().__init__(
            domain,
            grid_type,
            physical_constants=physical_constants,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        # instantiate the class computing the diagnostic variables
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = self.get_storage_shape(storage_shape, (nx, ny, nz + 1))
        self._core = Core(
            self.grid,
            physical_constants=physical_constants,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

        # allocate the storages collecting the output fields calculated
        # by the stencils
        self._out_p = self.zeros(shape=storage_shape)
        self._out_exn = self.zeros(shape=storage_shape)
        self._out_mtg = self.zeros(shape=storage_shape)
        self._out_h = self.zeros(shape=storage_shape)
        if moist:
            self._out_r = self.zeros(shape=storage_shape)
            self._out_t = self.zeros(shape=storage_shape)

    @property
    def input_properties(self) -> typing.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"}
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> typing.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
        dims_stgz = (dims[0], dims[1], self.grid.z_on_interface_levels.dims[0])

        return_dict = {
            "air_pressure_on_interface_levels": {
                "dims": dims_stgz,
                "units": "Pa",
            },
            "exner_function_on_interface_levels": {
                "dims": dims_stgz,
                "units": "J K^-1 kg^-1",
            },
            "height_on_interface_levels": {"dims": dims_stgz, "units": "m"},
            "montgomery_potential": {"dims": dims, "units": "m^2 s^-2"},
        }

        if self._moist:
            return_dict["air_density"] = {"dims": dims, "units": "kg m^-3"}
            return_dict["air_temperature"] = {"dims": dims, "units": "K"}

        return return_dict

    def array_call(self, state: typing.array_dict_t) -> typing.array_dict_t:
        s = state["air_isentropic_density"]
        self._core.get_diagnostic_variables(
            s, self._pt, self._out_p, self._out_exn, self._out_mtg, self._out_h
        )
        diagnostics = {
            "air_pressure_on_interface_levels": self._out_p,
            "exner_function_on_interface_levels": self._out_exn,
            "montgomery_potential": self._out_mtg,
            "height_on_interface_levels": self._out_h,
        }
        if self._moist:
            self._core.get_density_and_temperature(
                s, self._out_exn, self._out_h, self._out_r, self._out_t
            )
            diagnostics["air_density"] = self._out_r
            diagnostics["air_temperature"] = self._out_t

        return diagnostics


class IsentropicVelocityComponents(DiagnosticComponent):
    """
    Retrieve the horizontal velocity components with the help of the
    isentropic momenta and the isentropic density.
    The class is instantiated over the *numerical* grid of the
    underlying domain.
    """

    def __init__(
        self,
        domain: "Domain",
        *,
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
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        # initialize the parent class
        super().__init__(
            domain,
            "numerical",
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        # instantiate the class retrieving the velocity components
        self._core = HorizontalVelocity(
            self.grid,
            staggering=True,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        # set the shape of the storages
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = self.get_storage_shape(
            storage_shape, (nx + 1, ny + 1, nz)
        )

        # allocate the storages gathering the output fields
        self._out_u = self.zeros(shape=storage_shape)
        self._out_v = self.zeros(shape=storage_shape)

    @property
    def input_properties(self) -> typing.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> typing.properties_dict_t:
        g = self.grid
        dims_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
        dims_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

        return_dict = {
            "x_velocity_at_u_locations": {"dims": dims_x, "units": "m s^-1"},
            "y_velocity_at_v_locations": {"dims": dims_y, "units": "m s^-1"},
        }

        return return_dict

    def array_call(
        self, state: typing.gtstorage_dict_t
    ) -> typing.gtstorage_dict_t:
        # extract the required model variables from the input state
        s = state["air_isentropic_density"]
        su = state["x_momentum_isentropic"]
        sv = state["y_momentum_isentropic"]

        # diagnose the velocity components
        self._core.get_velocity_components(s, su, sv, self._out_u, self._out_v)

        # enforce the boundary conditions
        hb = self.horizontal_boundary
        hb.dmn_set_outermost_layers_x(
            self._out_u,
            field_name="x_velocity_at_u_locations",
            field_units="m s^-1",
            time=state["time"],
        )
        hb.dmn_set_outermost_layers_y(
            self._out_v,
            field_name="y_velocity_at_v_locations",
            field_units="m s^-1",
            time=state["time"],
        )

        # instantiate the output dictionary
        diagnostics = {
            "x_velocity_at_u_locations": self._out_u,
            "y_velocity_at_v_locations": self._out_v,
        }

        return diagnostics
