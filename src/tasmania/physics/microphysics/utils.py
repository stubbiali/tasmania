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

from __future__ import annotations
import abc
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript
from sympl._core.data_array import DataArray
from sympl._core.factory import AbstractFactory
from sympl._core.time import Timer

from tasmania.framework.core_components import DiagnosticComponent, ImplicitTendencyComponent
from tasmania.framework.stencil import StencilFactory
from tasmania.framework.tag import stencil_definition, subroutine_definition

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional

    from tasmania.domain.domain import Domain
    from tasmania.framework.options import BackendOptions, StorageOptions
    from tasmania.utils.typingx import (
        GTField,
        NDArray,
        NDArrayDict,
        PropertyDict,
        TimeDelta,
        TripletInt,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class Clipping(DiagnosticComponent):
    """Clipping negative values of water species."""

    def __init__(
        self,
        domain: Domain,
        grid_type: str,
        water_species_names: Optional[Sequence[str]] = None,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : str
            The type of grid over which instantiating the class.
            Either "physical" or "numerical".
        water_species_names : `tuple`, optional
            The names of the water species to clip.
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
        self._names = water_species_names
        super().__init__(
            domain,
            grid_type,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile_stencil("clip")

    @property
    def input_properties(self) -> PropertyDict:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def diagnostic_properties(self) -> PropertyDict:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    def array_call(self, state: NDArrayDict, out: NDArrayDict) -> None:
        for name in self._names:
            in_q = state[name]
            out_q = out[name]
            Timer.start(label="stencil")
            self._stencil(
                in_field=in_q,
                out_field=out_q,
                origin=(0, 0, 0),
                domain=out_q.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()


class Precipitation(ImplicitTendencyComponent):
    """Update the (accumulated) precipitation."""

    default_physical_constants = {
        "density_of_liquid_water": DataArray(1e3, attrs={"units": "kg m^-3"})
    }

    def __init__(
        self,
        domain: Domain,
        grid_type: str = "numerical",
        physical_constants: Optional[dict[str, DataArray]] = None,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional[StorageOptions] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        physical_constants : `dict[str, sympl.DataArray]`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'density_of_liquid_water', in units compatible with [kg m^-3].

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
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`~tasmania.ImplicitTendencyComponent`.
        """
        super().__init__(
            domain,
            grid_type,
            physical_constants=physical_constants,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
            **kwargs,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {"rhow": self.rpc["density_of_liquid_water"]}
        self._stencil = self.compile_stencil("accumulated_precipitation")

    @property
    def input_properties(self) -> PropertyDict:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims2d = (
            (g.x.dims[0], g.y.dims[0], g.z.dims[0] + "_at_surface_level")
            if g.nz > 1
            else (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        )

        return {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "mass_fraction_of_precipitation_water_in_air": {"dims": dims, "units": "g g^-1"},
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
            "accumulated_precipitation": {"dims": dims2d, "units": "mm"},
        }

    @property
    def tendency_properties(self) -> PropertyDict:
        return {}

    @property
    def diagnostic_properties(self) -> PropertyDict:
        g = self.grid
        dims2d = (
            (g.x.dims[0], g.y.dims[0], g.z.dims[0] + "_at_surface_level")
            if g.nz > 1
            else (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        )

        return {
            "precipitation": {"dims": dims2d, "units": "mm hr^-1"},
            "accumulated_precipitation": {"dims": dims2d, "units": "mm"},
        }

    def get_field_grid_shape(self, name: str) -> TripletInt:
        return self.grid.nx, self.grid.ny, 1

    def get_storage_shape(
        self,
        shape: Sequence[int],
        min_shape: Optional[Sequence[int]] = None,
        max_shape: Optional[Sequence[int]] = None,
    ) -> Sequence[int]:
        if shape is not None:
            shape = (shape[0], shape[1], shape[2] if self.grid.nz == 1 else 1)
        min_shape = min_shape or (self.grid.nx, self.grid.ny, 1)
        return self.get_shape(shape, min_shape, max_shape)

    def array_call(
        self,
        state: NDArrayDict,
        timestep: TimeDelta,
        out_tendencies: NDArrayDict,
        out_diagnostics: NDArrayDict,
        overwrite_tendencies: dict[str, bool],
    ) -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dt = timestep.total_seconds()
        Timer.start(label="stencil")
        self._stencil(
            in_rho=state["air_density"][:, :, nz - 1 : nz],
            in_qr=state[mfpw][:, :, nz - 1 : nz],
            in_vt=state["raindrop_fall_velocity"][:, :, nz - 1 : nz],
            in_accprec=state["accumulated_precipitation"][:, :, :1],
            out_prec=out_diagnostics["precipitation"][:, :, :1],
            out_accprec=out_diagnostics["accumulated_precipitation"][:, :, :1],
            dt=dt,
            origin=(0, 0, 0),
            domain=(nx, ny, 1),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="accumulated_precipitation"
    )
    def _accumulated_precipitation_numpy(
        in_rho: NDArray,
        in_qr: NDArray,
        in_vt: NDArray,
        in_accprec: NDArray,
        out_prec: NDArray,
        out_accprec: NDArray,
        *,
        dt: float,
        origin: TripletInt,
        domain: TripletInt,
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_prec[i, j, k] = 3.6e6 * in_rho[i, j, k] * in_qr[i, j, k] * in_vt[i, j, k] / rhow
        out_accprec[i, j, k] = in_accprec[i, j, k] + dt * out_prec[i, j, k] / 3.6e3

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="accumulated_precipitation")
    def _accumulated_precipitation_gt4py(
        in_rho: gtscript.Field["dtype"],
        in_qr: gtscript.Field["dtype"],
        in_vt: gtscript.Field["dtype"],
        in_accprec: gtscript.Field["dtype"],
        out_prec: gtscript.Field["dtype"],
        out_accprec: gtscript.Field["dtype"],
        *,
        dt: float,
    ) -> None:
        from __externals__ import rhow

        with computation(PARALLEL), interval(...):
            out_prec = 3.6e6 * in_rho * in_qr * in_vt / rhow
            out_accprec = in_accprec + dt * out_prec / 3.6e3


class SedimentationFlux(AbstractFactory, StencilFactory):
    """
    Abstract base class whose derived classes discretize the
    vertical derivative of the sedimentation flux with different
    orders of accuracy.
    """

    # the vertical extent of the stencil
    nb: int = None

    def __init__(self, *, backend: str = "numpy"):
        super().__init__(backend)

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux")
    @abc.abstractmethod
    def call_numpy(rho: NDArray, h: NDArray, q: NDArray, vt: NDArray) -> NDArray:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux")
    @gtscript.function
    @abc.abstractmethod
    def call_gt4py(rho: GTField, h: GTField, q: GTField, vt: GTField) -> GTField:
        """
        Get the vertical derivative of the sedimentation flux.
        As this method is marked as abstract, its implementation
        is delegated to the derived classes.

        Parameters
        ----------
        rho : gt4py.gtscript.Field
            The air density, in units of [kg m^-3].
        h : gt4py.gtscript.Field
            The geometric height of the model half-levels, in units of [m].
        q : gt4py.gtscript.Field
            The precipitating water species.
        vt : gt4py.gtscript.Field
            The raindrop fall velocity, in units of [m s^-1].

        Return
        ------
        gt4py.gtscript.Field :
            The vertical derivative of the sedimentation flux.
        """
