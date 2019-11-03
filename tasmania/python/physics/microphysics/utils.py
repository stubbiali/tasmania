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

import gridtools as gt
from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
)
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import get_storage_shape, zeros

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class Clipping(DiagnosticComponent):
    """
    Clipping negative values of water species.
    """

    def __init__(self, domain, grid_type, water_species_names=None):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : str
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical'.

        water_species_names : `tuple`, optional
            The names of the water species to clip.
        """
        self._names = water_species_names
        super().__init__(domain, grid_type)

    @property
    def input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def diagnostic_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {}
        for name in self._names:
            return_dict[name] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    def array_call(self, state):
        diagnostics = {}

        for name in self._names:
            q = state[name]
            q[q < 0.0] = 0.0
            diagnostics[name] = q

        return diagnostics


class Precipitation(ImplicitTendencyComponent):
    """
    Update the (accumulated) precipitation.
    """

    _d_physical_constants = {
        "density_of_liquid_water": DataArray(1e3, attrs={"units": "kg m^-3"})
    }

    def __init__(
        self,
        domain,
        grid_type="numerical",
        physical_constants=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        physical_constants : `dict`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'density_of_liquid_water', in units compatible with [kg m^-3].

        backend : `str`, optional
            TODO
        backend_opts : `dict`, optional
            TODO
        build_info : `dict`, optional
            TODO
        dtype : `numpy.dtype`, optional
            TODO
        exec_info : `dict`, optional
            TODO
        default_origin : `tuple`, optional
            TODO
        rebuild : `bool`, optional
            TODO
        storage_shape : `tuple`, optional
            TODO
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`~tasmania.ImplicitTendencyComponent`.
        """
        self._exec_info = exec_info

        super().__init__(domain, grid_type, **kwargs)

        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._pcs = pcs  # needed by unit test

        nx, ny = self.grid.nx, self.grid.ny
        in_shape = (
            (storage_shape[0], storage_shape[1], 1) if storage_shape is not None else None
        )
        storage_shape = get_storage_shape(in_shape, (nx, ny, 1))

        self._in_rho = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._in_qr = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._in_vt = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._out_prec = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )
        self._out_accprec = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )

        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            externals={"rhow": pcs["density_of_liquid_water"]},
            rebuild=rebuild,
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
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
    def tendency_properties(self):
        return {}

    @property
    def diagnostic_properties(self):
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

    def array_call(self, state, timestep):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # try:
        #     state["air_density"].host_to_device()
        #     self._in_rho.data[...] = state["air_density"].data[:, :, nz - 1 : nz]
        #     self._in_rho._sync_state.state = self._in_rho.SyncState.SYNC_DEVICE_DIRTY
        #
        #     state["mass_fraction_of_precipitation_water_in_air"].host_to_device()
        #     self._in_qr.data[...] = state[
        #         "mass_fraction_of_precipitation_water_in_air"
        #     ].data[:, :, nz - 1 : nz]
        #     self._in_qr._sync_state.state = self._in_qr.SyncState.SYNC_DEVICE_DIRTY
        #
        #     state["raindrop_fall_velocity"].host_to_device()
        #     self._in_vt.data[...] = state["raindrop_fall_velocity"].data[
        #         :, :, nz - 1 : nz
        #     ]
        #     self._in_vt._sync_state.state = self._in_vt.SyncState.SYNC_DEVICE_DIRTY
        # except AttributeError:
        #     self._in_rho[...] = state["air_density"][:, :, nz - 1 : nz]
        #     self._in_qr[...] = state["mass_fraction_of_precipitation_water_in_air"][
        #         :, :, nz - 1 : nz
        #     ]
        #     self._in_vt[...] = state["raindrop_fall_velocity"][:, :, nz - 1 : nz]

        self._in_rho[...] = state["air_density"][:, :, nz - 1 : nz]
        self._in_qr[...] = state["mass_fraction_of_precipitation_water_in_air"][
            :, :, nz - 1 : nz
        ]
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
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, 1),
            exec_info=self._exec_info,
        )

        tendencies = {}
        diagnostics = {
            "precipitation": self._out_prec,
            "accumulated_precipitation": self._out_accprec,
        }

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        in_vt: gt.storage.f64_sd,
        in_accprec: gt.storage.f64_sd,
        out_prec: gt.storage.f64_sd,
        out_accprec: gt.storage.f64_sd,
        *,
        dt: float
    ):
        out_prec = 3.6e6 * in_rho[0, 0, 0] * in_qr[0, 0, 0] * in_vt[0, 0, 0] / rhow
        out_accprec = in_accprec[0, 0, 0] + dt * out_prec[0, 0, 0] / 3.6e3


class SedimentationFlux(abc.ABC):
    """
    Abstract base class whose derived classes discretize the
    vertical derivative of the sedimentation flux with different
    orders of accuracy.
    """

    # the vertical extent of the stencil
    nb = None

    @staticmethod
    @abc.abstractmethod
    def __call__(rho, h, q, vt):
        """
        Get the vertical derivative of the sedimentation flux.
        As this method is marked as abstract, its implementation
        is delegated to the derived classes.

        Parameters
        ----------
        rho : gridtools.storage.Storage
            The air density, in units of [kg m^-3].
        h : gridtools.storage.Storage
            The geometric height of the model half-levels, in units of [m].
        q : gridtools.storage.Storage
            The precipitating water species.
        vt : gridtools.storage.Storage
            The raindrop fall velocity, in units of [m s^-1].

        Return
        ------
        gridtools.storage.Storage :
            The vertical derivative of the sedimentation flux.
        """

    @staticmethod
    def factory(sedimentation_flux_type):
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

        Return
        ------
            Instance of the derived class implementing the desired method.
        """
        if sedimentation_flux_type == "first_order_upwind":
            return _FirstOrderUpwind()
        elif sedimentation_flux_type == "second_order_upwind":
            return _SecondOrderUpwind()
        else:
            raise ValueError(
                "Only first- and second-order upwind methods have been implemented."
            )


class _FirstOrderUpwind(SedimentationFlux):
    """ The standard, first-order accurate upwind method. """

    nb = 1

    @staticmethod
    def __call__(rho, h, q, vt):
        # interpolate the geometric height at the model main levels
        tmp_h = 0.5 * (h[0, 0, 0] + h[0, 0, 1])

        # calculate the vertical derivative of the sedimentation flux
        dfdz = (
            rho[0, 0, -1] * q[0, 0, -1] * vt[0, 0, -1]
            - rho[0, 0, 0] * q[0, 0, 0] * vt[0, 0, 0]
        ) / (tmp_h[0, 0, -1] - tmp_h[0, 0, 0])

        return dfdz


class _SecondOrderUpwind(SedimentationFlux):
    """ The second-order accurate upwind method. """

    nb = 2

    @staticmethod
    def __call__(rho, h, q, vt):
        # interpolate the geometric height at the model main levels
        tmp_h = 0.5 * (h[0, 0, 0] + h[0, 0, 1])

        # evaluate the space-dependent coefficients occurring in the
        # second-order upwind finite difference approximation of the
        # vertical derivative of the flux
        tmp_a = (2.0 * tmp_h[0, 0, 0] - tmp_h[0, 0, -1] - tmp_h[0, 0, -2]) / (
            (tmp_h[0, 0, -1] - tmp_h[0, 0, 0]) * (tmp_h[0, 0, -2] - tmp_h[0, 0, 0])
        )
        tmp_b = (tmp_h[0, 0, -2] - tmp_h[0, 0, 0]) / (
            (tmp_h[0, 0, -1] - tmp_h[0, 0, 0]) * (tmp_h[0, 0, -2] - tmp_h[0, 0, -1])
        )
        tmp_c = (tmp_h[0, 0, 0] - tmp_h[0, 0, -1]) / (
            (tmp_h[0, 0, -2] - tmp_h[0, 0, 0]) * (tmp_h[0, 0, -2] - tmp_h[0, 0, -1])
        )

        # calculate the vertical derivative of the sedimentation flux
        dfdz = (
            tmp_a[0, 0, 0] * rho[0, 0, 0] * q[0, 0, 0] * vt[0, 0, 0]
            + tmp_b[0, 0, 0] * rho[0, 0, -1] * q[0, 0, -1] * vt[0, 0, -1]
            + tmp_c[0, 0, 0] * rho[0, 0, -2] * q[0, 0, -2] * vt[0, 0, -2]
        )

        return dfdz


class Sedimentation(ImplicitTendencyComponent):
    """
    Calculate the vertical derivative of the sedimentation flux for multiple
    precipitating tracers.
    """

    def __init__(
        self,
        domain,
        grid_type,
        tracers,
        sedimentation_flux_scheme="first_order_upwind",
        maximum_vertical_cfl=0.975,
        backend="numpy",
        dtype=datatype,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : str
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical'.

        tracers : dict
            Dictionary whose keys are the names of the precipitating tracers to
            consider, and whose values are dictionaries specifying 'units' and
            'velocity' for those tracers.
        sedimentation_flux_scheme : `str`, optional
            The numerical sedimentation flux scheme. Please refer to
            :class:`~tasmania.SedimentationFlux` for the available options.
            Defaults to 'first_order_upwind'.
        maximum_vertical_cfl : `float`, optional
            Maximum allowed vertical CFL number. Defaults to 0.975.
        backend : `obj`, optional
            TODO
        dtype : `numpy.dtype`, optional
            The data type for any :class:`numpy.ndarray` instantiated and
            used within this class.
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`~tasmania.ImplicitTendencyComponent`.
        """
        self._tracer_units = {}
        self._velocities = {}
        for tracer in tracers:
            try:
                self._tracer_units[tracer] = tracers[tracer]["units"]
            except KeyError:
                raise KeyError(
                    "Dictionary for " "{}" " misses the key " "units" ".".format(tracer)
                )

            try:
                self._velocities[tracer] = tracers[tracer]["velocity"]
            except KeyError:
                raise KeyError(
                    "Dictionary for "
                    "{}"
                    " misses the key "
                    "velocity"
                    ".".format(tracer)
                )

        super().__init__(domain, grid_type, **kwargs)

        self._sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
        self._max_cfl = maximum_vertical_cfl
        self._stencil_initialize(backend, dtype)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "height_on_interface_levels": {"dims": dims_z, "units": "m"},
        }

        for tracer in self._tracer_units:
            return_dict[tracer] = {"dims": dims, "units": self._tracer_units[tracer]}
            return_dict[self._velocities[tracer]] = {"dims": dims, "units": "m s^-1"}

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {}
        for tracer, units in self._tracer_units.items():
            return_dict[tracer] = {"dims": dims, "units": units + " s^-1"}

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state, timestep):
        self._stencil_set_inputs(state, timestep)

        self._stencil.compute()

        tendencies = {name: self._outputs["out_" + name] for name in self._tracer_units}
        diagnostics = {}

        return tendencies, diagnostics

    def _stencil_initialize(self, backend, dtype):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        self._dt = gt.Global()
        self._maxcfl = gt.Global(self._max_cfl)

        self._inputs = {
            "in_rho": np.zeros((nx, ny, nz), dtype=dtype),
            "in_h": np.zeros((nx, ny, nz + 1), dtype=dtype),
        }
        self._outputs = {}
        for tracer in self._tracer_units:
            self._inputs["in_" + tracer] = np.zeros((nx, ny, nz), dtype=dtype)
            self._inputs["in_" + self._velocities[tracer]] = np.zeros(
                (nx, ny, nz), dtype=dtype
            )
            self._outputs["out_" + tracer] = np.zeros((nx, ny, nz), dtype=dtype)

        self._stencil = gt.NGStencil(
            definitions_func=self._stencil_defs,
            inputs=self._inputs,
            global_inputs={"dt": self._dt, "max_cfl": self._maxcfl},
            outputs=self._outputs,
            domain=gt.domain.Rectangle((0, 0, self._sflux.nb), (nx - 1, ny - 1, nz - 1)),
            mode=backend,
        )

    def _stencil_set_inputs(self, state, timestep):
        self._dt.value = timestep.total_seconds()
        self._inputs["in_rho"][...] = state["air_density"][...]
        self._inputs["in_h"][...] = state["height_on_interface_levels"][...]
        for tracer in self._tracer_units:
            self._inputs["in_" + tracer][...] = state[tracer][...]
            velocity = self._velocities[tracer]
            self._inputs["in_" + velocity] = state[velocity][...]

    def _stencil_defs(self, dt, max_cfl, in_rho, in_h, **kwargs):
        k = gt.Index(axis=2)

        tmp_dh = gt.Equation()
        tmp_dh[k] = in_h[k] - in_h[k + 1]

        outs = []

        for tracer in self._tracer_units:
            in_q = kwargs["in_" + tracer]
            in_vt = kwargs["in_" + self._velocities[tracer]]

            tmp_vt = gt.Equation(name="tmp_" + self._velocities[tracer])
            tmp_vt[k] = in_vt[k]
            # 	(vt[k] >  max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
            # 	(vt[k] <= max_cfl * tmp_dh[k] / dt) * vt[k]

            tmp_dfdz = gt.Equation(name="tmp_dfdz_" + tracer)
            self._sflux(k, in_rho, in_h, in_q, tmp_vt, tmp_dfdz)

            out_q = gt.Equation(name="out_" + tracer)
            out_q[k] = tmp_dfdz[k] / in_rho[k]

            outs.append(out_q)

        return outs
