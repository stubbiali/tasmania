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
from sympl import DataArray
from typing import Mapping, Optional, TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import get_storage_shape, zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


class DryStaticEnergy(DiagnosticComponent):
    """ Diagnose the dry static energy. """

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "gravitational_acceleration": DataArray(9.80665, attrs={"units": "m s^-2"}),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        height_on_interface_levels: bool = True,
        physical_constants: Optional[Mapping[str, DataArray]] = None,
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
        # keep track of input arguments needed at run-time
        self._stgz = height_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type)

        # set physical parameters values
        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._g = pcs["gravitational_acceleration"]  # debug purposes
        self._cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]  # debug purposes

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz))

        # allocate the gt4py storage storing the output
        self._out_dse = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        if gt_powered:
            # instantiate the underlying gt4py stencil object
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals={
                    "height_on_interface_levels": self._stgz,
                    "g": pcs["gravitational_acceleration"],
                    "cp": pcs["specific_heat_of_dry_air_at_constant_pressure"],
                },
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {"air_temperature": {"dims": dims, "units": "K"}}
        if self._stgz:
            return_dict["height_on_interface_levels"] = {"dims": dims_stgz, "units": "m"}
        else:
            return_dict["height"] = {"dims": dims, "units": "m"}

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {"montgomery_potential": {"dims": dims, "units": "m^2 s^-2"}}

        return return_dict

    def array_call(self, state: taz_types.array_dict_t) -> taz_types.array_dict_t:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_t = state["air_temperature"]
        in_h = state["height_on_interface_levels"] if self._stgz else state["height"]
        out_dse = self._out_dse

        self._stencil(
            in_t=in_t,
            in_h=in_h,
            out_dse=out_dse,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        diagnostics = {"montgomery_potential": out_dse}

        return diagnostics

    def _stencil_numpy(
        self,
        in_t: np.ndarray,
        in_h: np.ndarray,
        out_dse: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])
        kp1 = slice(origin[2] + 1, origin[2] + domain[2] + 1)

        if self._stgz:
            out_dse[i, j, k] = self._cp * in_t[i, j, k] + self._g * 0.5 * (
                in_h[i, j, k] + in_h[i, j, kp1]
            )
        else:
            out_dse[i, j, k] = self._cp * in_t[i, j, k] + self._g * in_h[i, j, k]

    @staticmethod
    def _stencil_gt_defs(
        in_t: gtscript.Field["dtype"],
        in_h: gtscript.Field["dtype"],
        out_dse: gtscript.Field["dtype"],
    ):
        from __externals__ import cp, g, height_on_interface_levels

        with computation(PARALLEL), interval(...):
            if __INLINED(height_on_interface_levels):
                h = 0.5 * (in_h[0, 0, 0] + in_h[0, 0, 1])
            else:
                h = in_h

            out_dse = cp * in_t + g * h


class MoistStaticEnergy(DiagnosticComponent):
    """ Diagnose the moist static energy. """

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        )
    }

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
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
        managed_memory: bool = False,
    ) -> None:
        # keep track of input arguments needed at run-time
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type)

        # set physical parameters values
        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._lhvw = pcs["latent_heat_of_vaporization_of_water"]

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz))

        # allocate the gt4py storage storing the output
        self._out_mse = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        if gt_powered:
            # instantiate the underlying gt4py stencil object
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals={"lhvw": pcs["latent_heat_of_vaporization_of_water"]},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "montgomery_potential": {"dims": dims, "units": "m^2 s^-2"},
            "mass_fraction_of_water_vapor_in_air": {"dims": dims, "units": "g g^-1"},
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {"moist_static_energy": {"dims": dims, "units": "m^2 s^-2"}}

        return return_dict

    def array_call(self, state: taz_types.array_dict_t) -> taz_types.array_dict_t:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_dse = state["montgomery_potential"]
        in_qv = state["mass_fraction_of_water_vapor_in_air"]
        out_mse = self._out_mse

        self._stencil(
            in_dse=in_dse,
            in_qv=in_qv,
            out_mse=out_mse,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        diagnostics = {"moist_static_energy": out_mse}

        return diagnostics

    def _stencil_numpy(
        self,
        in_dse: np.ndarray,
        in_qv: np.ndarray,
        out_mse: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_mse[i, j, k] = in_dse[i, j, k] + self._lhvw * in_qv[i, j, k]

    @staticmethod
    def _stencil_gt_defs(
        in_dse: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"],
        out_mse: gtscript.Field["dtype"],
    ):
        from __externals__ import lhvw

        with computation(PARALLEL), interval(...):
            out_mse = in_dse + lhvw * in_qv
