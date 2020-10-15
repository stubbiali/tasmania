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
import abc
import numpy as np
from sympl import DataArray
from typing import Dict, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple

from gt4py import gtscript

from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
)
from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.framework_utils import factorize
from tasmania.python.utils.gtscript_utils import stencil_clip_defs
from tasmania.python.utils.storage_utils import get_storage_shape, zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class Clipping(DiagnosticComponent):
    """ Clipping negative values of water species. """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        water_species_names: Optional[Sequence[str]] = None,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
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
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : str
            The type of grid over which instantiating the class.
            Either "physical" or "numerical".
        water_species_names : `tuple`, optional
            The names of the water species to clip.
        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        """
        self._names = water_species_names
        super().__init__(domain, grid_type)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        in_shape = storage_shape or None
        storage_shape = get_storage_shape(in_shape, (nx, ny, nz))
        self._outs = {
            name: zeros(
                storage_shape,
                backend=backend,
                dtype=dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )
            for name in water_species_names
        }

        if is_gt(backend):
            self._stencil = gtscript.stencil(
                backend=get_gt_backend(backend),
                definition=stencil_clip_defs,
                rebuild=rebuild,
                build_info=build_info,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    def array_call(
        self, state: taz_types.array_dict_t
    ) -> taz_types.array_dict_t:
        diagnostics = {}

        for name in self._names:
            in_q = state[name]
            out_q = self._outs[name]
            self._stencil(
                in_field=in_q,
                out_field=out_q,
                origin=(0, 0, 0),
                domain=out_q.shape,
                validate_args=False,
            )
            diagnostics[name] = out_q

        return diagnostics

    @staticmethod
    def _stencil_numpy(
        in_field: np.ndarray,
        out_field: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_field[i, j, k] = np.where(
            in_field[i, j, k] > 0.0, in_field[i, j, k], 0.0
        )


class Precipitation(ImplicitTendencyComponent):
    """ Update the (accumulated) precipitation. """

    _d_physical_constants = {
        "density_of_liquid_water": DataArray(1e3, attrs={"units": "kg m^-3"})
    }

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False,
        **kwargs
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

        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`~tasmania.ImplicitTendencyComponent`.
        """
        self._exec_info = exec_info

        super().__init__(domain, grid_type, **kwargs)

        self._pcs = get_physical_constants(
            self._d_physical_constants, physical_constants
        )

        nx, ny = self.grid.nx, self.grid.ny
        in_shape = (
            (storage_shape[0], storage_shape[1], 1)
            if storage_shape is not None
            else None
        )
        storage_shape = get_storage_shape(in_shape, (nx, ny, 1))

        self._in_rho = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._in_qr = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._in_vt = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_prec = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_accprec = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        if is_gt(backend):
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                dtypes={"dtype": dtype},
                externals={"rhow": self._pcs["density_of_liquid_water"]},
                rebuild=rebuild,
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims2d = (
            (g.x.dims[0], g.y.dims[0], g.z.dims[0] + "_at_surface_level")
            if g.nz > 1
            else (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        )

        return {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
            "accumulated_precipitation": {"dims": dims2d, "units": "mm"},
        }

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
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

    def array_call(
        self, state: taz_types.array_dict_t, timestep: taz_types.timedelta_t
    ) -> Tuple[taz_types.array_dict_t, taz_types.array_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        self._in_rho[...] = state["air_density"][:, :, nz - 1 : nz]
        self._in_qr[...] = state[
            "mass_fraction_of_precipitation_water_in_air"
        ][:, :, nz - 1 : nz]
        self._in_vt[...] = state["raindrop_fall_velocity"][:, :, nz - 1 : nz]
        in_accprec = state["accumulated_precipitation"]

        dt = timestep.total_seconds()

        self._stencil(
            in_rho=self._in_rho,
            in_qr=self._in_qr,
            in_vt=self._in_vt,
            in_accprec=in_accprec,
            out_prec=self._out_prec,
            out_accprec=self._out_accprec,
            dt=dt,
            origin=(0, 0, 0),
            domain=(nx, ny, 1),
            exec_info=self._exec_info,
            validate_args=False,
        )

        tendencies = {}
        diagnostics = {
            "precipitation": self._out_prec,
            "accumulated_precipitation": self._out_accprec,
        }

        return tendencies, diagnostics

    def _stencil_numpy(
        self,
        in_rho: np.ndarray,
        in_qr: np.ndarray,
        in_vt: np.ndarray,
        in_accprec: np.ndarray,
        out_prec: np.ndarray,
        out_accprec: np.ndarray,
        *,
        dt: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_prec[i, j, k] = (
            3.6e6
            * in_rho[i, j, k]
            * in_qr[i, j, k]
            * in_vt[i, j, k]
            / self._pcs["density_of_liquid_water"]
        )
        out_accprec[i, j, k] = (
            in_accprec[i, j, k] + dt * out_prec[i, j, k] / 3.6e3
        )

    @staticmethod
    def _stencil_gt_defs(
        in_rho: gtscript.Field["dtype"],
        in_qr: gtscript.Field["dtype"],
        in_vt: gtscript.Field["dtype"],
        in_accprec: gtscript.Field["dtype"],
        out_prec: gtscript.Field["dtype"],
        out_accprec: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        from __externals__ import rhow

        with computation(PARALLEL), interval(...):
            out_prec = 3.6e6 * in_rho * in_qr * in_vt / rhow
            out_accprec = in_accprec + dt * out_prec / 3.6e3


class SedimentationFlux(abc.ABC):
    """
    Abstract base class whose derived classes discretize the
    vertical derivative of the sedimentation flux with different
    orders of accuracy.
    """

    # the vertical extent of the stencil
    nb: int = None

    # registry of concrete subclasses
    registry: Dict[str, "SedimentationFlux"] = {}

    def __init__(self, backend: str) -> None:
        self.call = self.call_gt if is_gt(backend) else self.call_numpy

    @staticmethod
    @abc.abstractmethod
    def call_numpy(
        rho: np.ndarray, h: np.ndarray, q: np.ndarray, vt: np.ndarray
    ) -> np.ndarray:
        pass

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def call_gt(
        rho: taz_types.gtfield_t,
        h: taz_types.gtfield_t,
        q: taz_types.gtfield_t,
        vt: taz_types.gtfield_t,
    ) -> taz_types.gtfield_t:
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
        pass

    @staticmethod
    def factory(
        sedimentation_flux_type: str, backend: str
    ) -> "SedimentationFlux":
        """
        Static method returning an instance of the derived class
        which discretizes the vertical derivative of the
        sedimentation flux with the desired level of accuracy.

        Parameters
        ----------
        sedimentation_flux_type : str
            String specifying the method used to compute the numerical
            sedimentation flux. Available options are:

                - 'first_order_upwind', for the first-order upwind scheme;
                - 'second_order_upwind', for the second-order upwind scheme.

        backend: str
            The backend.

        Return
        ------
        obj :
            Instance of the derived class implementing the desired method.
        """
        return factorize(
            sedimentation_flux_type, SedimentationFlux, (backend,)
        )


# class Sedimentation(ImplicitTendencyComponent):
#     """
#     Calculate the vertical derivative of the sedimentation flux for multiple
#     precipitating tracers.
#     """
#
#     def __init__(
#         self,
#         domain,
#         grid_type,
#         tracers,
#         sedimentation_flux_scheme="first_order_upwind",
#         maximum_vertical_cfl=0.975,
#         backend="numpy",
#         dtype=np.float64,
#         **kwargs
#     ):
#         """
#         Parameters
#         ----------
#         domain : tasmania.Domain
#             The :class:`~tasmania.Domain` holding the grid underneath.
#         grid_type : str
#             The type of grid over which instantiating the class.
#             Either "physical" or "numerical".
#         tracers : dict[str, dict]
#             Dictionary whose keys are the names of the precipitating tracers to
#             consider, and whose values are dictionaries specifying 'units' and
#             'velocity' for those tracers.
#         sedimentation_flux_scheme : `str`, optional
#             The numerical sedimentation flux scheme. Please refer to
#             :class:`~tasmania.SedimentationFlux` for the available options.
#             Defaults to 'first_order_upwind'.
#         maximum_vertical_cfl : `float`, optional
#             Maximum allowed vertical CFL number. Defaults to 0.975.
#         backend : `str`, optional
#             The backend.
#         dtype : `data-type`, optional
#             The data type for any storage instantiated and used within this
#             class.
#         **kwargs :
#             Additional keyword arguments to be directly forwarded to the parent
#             :class:`~tasmania.ImplicitTendencyComponent`.
#         """
#         self._tracer_units = {}
#         self._velocities = {}
#         for tracer in tracers:
#             try:
#                 self._tracer_units[tracer] = tracers[tracer]["units"]
#             except KeyError:
#                 raise KeyError(
#                     "Dictionary for "
#                     "{}"
#                     " misses the key "
#                     "units"
#                     ".".format(tracer)
#                 )
#
#             try:
#                 self._velocities[tracer] = tracers[tracer]["velocity"]
#             except KeyError:
#                 raise KeyError(
#                     "Dictionary for "
#                     "{}"
#                     " misses the key "
#                     "velocity"
#                     ".".format(tracer)
#                 )
#
#         super().__init__(domain, grid_type, **kwargs)
#
#         self._sflux = SedimentationFlux.factory(
#             sedimentation_flux_scheme, backend
#         )
#         self._max_cfl = maximum_vertical_cfl
#         self._stencil_initialize(backend, dtype)
#
#     @property
#     def input_properties(self):
#         g = self.grid
#         dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
#         dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
#
#         return_dict = {
#             "air_density": {"dims": dims, "units": "kg m^-3"},
#             "height_on_interface_levels": {"dims": dims_z, "units": "m"},
#         }
#
#         for tracer in self._tracer_units:
#             return_dict[tracer] = {
#                 "dims": dims,
#                 "units": self._tracer_units[tracer],
#             }
#             return_dict[self._velocities[tracer]] = {
#                 "dims": dims,
#                 "units": "m s^-1",
#             }
#
#         return return_dict
#
#     @property
#     def tendency_properties(self):
#         g = self.grid
#         dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
#
#         return_dict = {}
#         for tracer, units in self._tracer_units.items():
#             return_dict[tracer] = {"dims": dims, "units": units + " s^-1"}
#
#         return return_dict
#
#     @property
#     def diagnostic_properties(self):
#         return {}
#
#     def array_call(self, state, timestep):
#         self._stencil_set_inputs(state, timestep)
#
#         self._stencil.compute()
#
#         tendencies = {
#             name: self._outputs["out_" + name] for name in self._tracer_units
#         }
#         diagnostics = {}
#
#         return tendencies, diagnostics
#
#     def _stencil_initialize(self, backend, dtype):
#         nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
#
#         self._dt = gt.Global()
#         self._maxcfl = gt.Global(self._max_cfl)
#
#         self._inputs = {
#             "in_rho": np.zeros((nx, ny, nz), dtype=dtype),
#             "in_h": np.zeros((nx, ny, nz + 1), dtype=dtype),
#         }
#         self._outputs = {}
#         for tracer in self._tracer_units:
#             self._inputs["in_" + tracer] = np.zeros((nx, ny, nz), dtype=dtype)
#             self._inputs["in_" + self._velocities[tracer]] = np.zeros(
#                 (nx, ny, nz), dtype=dtype
#             )
#             self._outputs["out_" + tracer] = np.zeros(
#                 (nx, ny, nz), dtype=dtype
#             )
#
#         self._stencil = gt.NGStencil(
#             definitions_func=self._stencil_gt_defs,
#             inputs=self._inputs,
#             global_inputs={"dt": self._dt, "max_cfl": self._maxcfl},
#             outputs=self._outputs,
#             domain=gt.domain.Rectangle(
#                 (0, 0, self._sflux.nb), (nx - 1, ny - 1, nz - 1)
#             ),
#             mode=backend,
#         )
#
#     def _stencil_set_inputs(self, state, timestep):
#         self._dt.value = timestep.total_seconds()
#         self._inputs["in_rho"][...] = state["air_density"][...]
#         self._inputs["in_h"][...] = state["height_on_interface_levels"][...]
#         for tracer in self._tracer_units:
#             self._inputs["in_" + tracer][...] = state[tracer][...]
#             velocity = self._velocities[tracer]
#             self._inputs["in_" + velocity] = state[velocity][...]
#
#     def _stencil_gt_defs(self, dt, max_cfl, in_rho, in_h, **kwargs):
#         k = gt.Index(axis=2)
#
#         tmp_dh = gt.Equation()
#         tmp_dh[k] = in_h[k] - in_h[k + 1]
#
#         outs = []
#
#         for tracer in self._tracer_units:
#             in_q = kwargs["in_" + tracer]
#             in_vt = kwargs["in_" + self._velocities[tracer]]
#
#             tmp_vt = gt.Equation(name="tmp_" + self._velocities[tracer])
#             tmp_vt[k] = in_vt[k]
#             # 	(vt[k] >  max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
#             # 	(vt[k] <= max_cfl * tmp_dh[k] / dt) * vt[k]
#
#             tmp_dfdz = gt.Equation(name="tmp_dfdz_" + tracer)
#             self._sflux(k, in_rho, in_h, in_q, tmp_vt, tmp_dfdz)
#
#             out_q = gt.Equation(name="out_" + tracer)
#             out_q[k] = tmp_dfdz[k] / in_rho[k]
#
#             outs.append(out_q)
#
#         return outs
