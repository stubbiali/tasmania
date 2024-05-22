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
import numpy as np
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

from sympl._core.data_array import DataArray
from sympl._core.time import Timer

from gt4py import gtscript

from tasmania.python.framework.core_components import DiagnosticComponent
from tasmania.python.framework.tag import stencil_definition

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TripletInt


class DryStaticEnergy(DiagnosticComponent):
    """Diagnose the dry static energy."""

    # default values for the physical constants used in the class
    default_physical_constants = {
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
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> None:
        # keep track of input arguments needed at run-time
        self._stgz = height_on_interface_levels

        # call parent's constructor
        super().__init__(
            domain,
            grid_type,
            physical_constants,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

        # set physical parameters values
        self._g = self.rpc["gravitational_acceleration"]  # debug purposes
        self._cp = self.rpc["specific_heat_of_dry_air_at_constant_pressure"]  # debug purposes

        # instantiate the underlying stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "height_on_interface_levels": self._stgz,
            "g": self.rpc["gravitational_acceleration"],
            "cp": self.rpc["specific_heat_of_dry_air_at_constant_pressure"],
        }
        self._stencil = self.compile_stencil("static_energy")

    @property
    def input_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {"air_temperature": {"dims": dims, "units": "K"}}
        if self._stgz:
            return_dict["height_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "m",
            }
        else:
            return_dict["height"] = {"dims": dims, "units": "m"}

        return return_dict

    @property
    def diagnostic_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {"montgomery_potential": {"dims": dims, "units": "m^2 s^-2"}}

        return return_dict

    def array_call(self, state: "NDArrayLikeDict", out: "NDArrayLikeDict") -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_t = state["air_temperature"]
        in_h = state["height_on_interface_levels"] if self._stgz else state["height"]
        out_dse = out["montgomery_potential"]

        Timer.start(label="stencil")
        self._stencil(
            in_t=in_t,
            in_h=in_h,
            out_dse=out_dse,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @stencil_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="static_energy")
    def _stencil_numpy(
        self,
        in_t: np.ndarray,
        in_h: np.ndarray,
        out_dse: np.ndarray,
        *,
        origin: "TripletInt",
        domain: "TripletInt",
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])
        kp1 = slice(origin[2] + 1, origin[2] + domain[2] + 1)

        if height_on_interface_levels:
            out_dse[i, j, k] = cp * in_t[i, j, k] + g * 0.5 * (in_h[i, j, k] + in_h[i, j, kp1])
        else:
            out_dse[i, j, k] = cp * in_t[i, j, k] + g * in_h[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="static_energy")
    def _stencil_gt4py(
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
    """Diagnose the moist static energy."""

    # default values for the physical constants used in the class
    default_physical_constants = {
        "latent_heat_of_vaporization_of_water": DataArray(2.5e6, attrs={"units": "J kg^-1"})
    }

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> None:
        # call parent's constructor
        super().__init__(
            domain,
            grid_type,
            physical_constants,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )

        # instantiate the underlying stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {"lhvw": self.rpc["latent_heat_of_vaporization_of_water"]}
        self._stencil = self.compile_stencil("moist_static_energy")

    @property
    def input_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "montgomery_potential": {"dims": dims, "units": "m^2 s^-2"},
            "mass_fraction_of_water_vapor_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> "PropertyDict":
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {"moist_static_energy": {"dims": dims, "units": "m^2 s^-2"}}

        return return_dict

    def array_call(self, state: "NDArrayLikeDict", out: "NDArrayLikeDict") -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_dse = state["montgomery_potential"]
        in_qv = state["mass_fraction_of_water_vapor_in_air"]
        out_mse = out["moist_static_energy"]

        Timer.start(label="stencil")
        self._stencil(
            in_dse=in_dse,
            in_qv=in_qv,
            out_mse=out_mse,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"),
        stencil="moist_static_energy",
    )
    def _stencil_numpy(
        self,
        in_dse: np.ndarray,
        in_qv: np.ndarray,
        out_mse: np.ndarray,
        *,
        origin: "TripletInt",
        domain: "TripletInt",
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_mse[i, j, k] = in_dse[i, j, k] + lhvw * in_qv[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="moist_static_energy")
    def _stencil_gt4py(
        in_dse: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"],
        out_mse: gtscript.Field["dtype"],
    ):
        from __externals__ import lhvw

        with computation(PARALLEL), interval(...):
            out_mse = in_dse + lhvw * in_qv
