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

import gridtools as gt
from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    TendencyComponent,
)
from tasmania.python.physics.microphysics.utils import SedimentationFlux
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import empty, get_storage_shape, zeros
from tasmania.python.utils.meteo_utils import goff_gratch_formula, tetens_formula

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class KesslerMicrophysics(TendencyComponent):
    """
    The WRF version of the Kessler microphysics scheme.

    Note
    ----
    The calculated tendencies do not include the source terms deriving
    from the saturation adjustment.

    References
    ----------
    Doms, G., et al. (2015). A description of the nonhydrostatic regional \
        COSMO-model. Part II: Physical parameterization. \
        Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
    Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
        Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
        Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
    """

    # default values for the physical parameters used in the class
    _d_a = DataArray(0.001, attrs={"units": "g g^-1"})
    _d_k1 = DataArray(0.001, attrs={"units": "s^-1"})
    _d_k2 = DataArray(2.2, attrs={"units": "s^-1"})

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
        "gas_constant_of_water_vapor": DataArray(461.52, attrs={"units": "J K^-1 kg^-1"}),
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        ),
    }

    def __init__(
        self,
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        rain_evaporation=True,
        autoconversion_threshold=_d_a,
        autoconversion_rate=_d_k1,
        collection_rate=_d_k2,
        saturation_water_vapor_formula="tetens",
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

        air_pressure_on_interface_levels : `bool`, optional
            :obj:`True` (respectively, :obj:`False`) if the input pressure
            field is defined at the interface (resp., main) levels.
            Defaults to :obj:`True`.
        tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
            :obj:`True` to include the tendency for the potential
            temperature in the output dictionary collecting the diagnostics,
            :obj:`False` otherwise. Defaults to :obj:`False`.
        rain_evaporation : `bool`, optional
            :obj:`True` if the evaporation of raindrops should be taken
            into account, :obj:`False` otherwise. Defaults to :obj:`True`.
        autoconversion_threshold : `sympl.DataArray`, optional
            Autoconversion threshold, in units compatible with [g g^-1].
        autoconversion_rate : `sympl.DataArray`, optional
            Autoconversion rate, in units compatible with [s^-1].
        collection_rate : `sympl.DataArray`, optional
            Rate of collection, in units compatible with [s^-1].
        saturation_water_vapor_formula : `str`, optional
            The formula giving the saturation water vapor. Available options are:

                * 'tetens' (default) for the Tetens' equation;
                * 'goff_gratch' for the Goff-Gratch equation.

        physical_constants : `dict`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'gas_constant_of_dry_air', in units compatible with \
                    [J K^-1 kg^-1];
                * 'gas_constant_of_water_vapor', in units compatible with \
                    [J K^-1 kg^-1];
                * 'latent_heat_of_vaporization_of_water', in units compatible with \
                    [J kg^-1].

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
            :class:`tasmania.TendencyComponent`.
        """
        # keep track of input arguments needed at run-time
        self._pttd = tendency_of_air_potential_temperature_in_diagnostics
        self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
        self._rain_evaporation = rain_evaporation
        self._a = autoconversion_threshold.to_units("g g^-1").values.item()
        self._k1 = autoconversion_rate.to_units("s^-1").values.item()
        self._k2 = collection_rate.to_units("s^-1").values.item()
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type, **kwargs)

        # set physical parameters values
        pcs = get_physical_constants(self._d_physical_constants, physical_constants)

        # set the formula calculating the saturation water vapor pressure
        self._swvf = (
            goff_gratch_formula
            if saturation_water_vapor_formula == "goff_gratch"
            else tetens_formula
        )

        # shortcuts
        rd = pcs["gas_constant_of_dry_air"]
        rv = pcs["gas_constant_of_water_vapor"]
        beta = rd / rv

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz + 1))

        # allocate the gt4py storage collecting the outputs
        self._in_ps = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._out_qc_tnd = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )
        self._out_qr_tnd = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )
        if rain_evaporation:
            self._out_qv_tnd = zeros(
                storage_shape, backend, dtype, default_origin=default_origin
            )
            self._out_theta_tnd = zeros(
                storage_shape, backend, dtype, default_origin=default_origin
            )

        # initialize the underlying gt4py stencil object
        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            min_signature=True,
            rebuild=rebuild,
            module="kessler_microphysics",
            externals={
                "air_pressure_on_interface_levels": air_pressure_on_interface_levels,
                "beta": beta,
                "lhvw": pcs["latent_heat_of_vaporization_of_water"],
                "rain_evaporation": rain_evaporation,
            },
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
        dims_on_interface_levels = (
            grid.x.dims[0],
            grid.y.dims[0],
            grid.z_on_interface_levels.dims[0],
        )

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "air_temperature": {"dims": dims, "units": "K"},
            "mass_fraction_of_water_vapor_in_air": {"dims": dims, "units": "g g^-1"},
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
        }

        if self._air_pressure_on_interface_levels:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "Pa",
            }
            return_dict["exner_function_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "J K^-1 kg^-1",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}
            return_dict["exner_function"] = {"dims": dims, "units": "J K^-1 kg^-1"}

        return return_dict

    @property
    def tendency_properties(self):
        grid = self._grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "g g^-1 s^-1",
            },
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1 s^-1",
            },
        }

        if self._rain_evaporation:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }

            if not self._pttd:
                return_dict["air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    @property
    def diagnostic_properties(self):
        if self._rain_evaporation and self._pttd:
            grid = self._grid
            dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
            return {
                "tendency_of_air_potential_temperature": {"dims": dims, "units": "K s^-1"}
            }
        else:
            return {}

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # extract the required model variables
        in_rho = state["air_density"]
        in_t = state["air_temperature"]
        in_qc = state[mfcw]
        in_qr = state[mfpw]
        if self._rain_evaporation:
            in_qv = state[mfwv]
        if self._air_pressure_on_interface_levels:
            in_p = state["air_pressure_on_interface_levels"]
            in_exn = state["exner_function_on_interface_levels"]
        else:
            in_p = state["air_pressure"]
            in_exn = state["exner_function"]

        # compute the saturation water vapor pressure
        try:
            in_t.host_to_device()
            self._in_ps.data[...] = self._swvf(in_t.data)
            self._in_ps._sync_state.state = self._in_ps.SyncState.SYNC_DEVICE_DIRTY
        except AttributeError:
            self._in_ps[...] = self._swvf(in_t)

        # collect the stencil arguments
        stencil_args = {
            "a": self._a,
            "k1": self._k1,
            "k2": self._k2,
            "in_rho": in_rho,
            "in_p": in_p,
            "in_ps": self._in_ps,
            "in_exn": in_exn,
            "in_qc": in_qc,
            "in_qr": in_qr,
            "out_qc_tnd": self._out_qc_tnd,
            "out_qr_tnd": self._out_qr_tnd,
        }
        if self._rain_evaporation:
            stencil_args["in_qv"] = in_qv
            stencil_args["out_qv_tnd"] = self._out_qv_tnd
            stencil_args["out_theta_tnd"] = self._out_theta_tnd

        # run the stencil
        self._stencil(**stencil_args, origin={"_all_": (0, 0, 0)}, domain=(nx, ny, nz))

        # collect the tendencies
        # >>> comment the following two lines before testing <<<
        # self._out_qc_tnd[np.isnan(self._out_qc_tnd)] = 0.0
        # self._out_qr_tnd[np.isnan(self._out_qr_tnd)] = 0.0
        tendencies = {mfcw: self._out_qc_tnd, mfpw: self._out_qr_tnd}
        if self._rain_evaporation:
            # >>> comment the following two lines before testing <<<
            # self._out_qv_tnd[np.isnan(self._out_qv_tnd)] = 0.0
            # self._out_theta_tnd[np.isnan(self._out_theta_tnd)] = 0.0
            tendencies[mfwv] = self._out_qv_tnd
            if not self._pttd:
                tendencies["air_potential_temperature"] = self._out_theta_tnd

        # collect the diagnostics
        if self._rain_evaporation and self._pttd:
            diagnostics = {"tendency_of_air_potential_temperature": self._out_theta_tnd}
        else:
            diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_p: gt.storage.f64_sd,
        in_ps: gt.storage.f64_sd,
        in_exn: gt.storage.f64_sd,
        in_qv: gt.storage.f64_sd,
        in_qc: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        out_qv_tnd: gt.storage.f64_sd,
        out_qc_tnd: gt.storage.f64_sd,
        out_qr_tnd: gt.storage.f64_sd,
        out_theta_tnd: gt.storage.f64_sd,
        *,
        a: float,
        k1: float,
        k2: float
    ):
        # interpolate the pressure and the Exner function at the vertical main levels
        if air_pressure_on_interface_levels:
            p = 0.5 * (in_p[0, 0, 0] + in_p[0, 0, 1])
            exn = 0.5 * (in_exn[0, 0, 0] + in_exn[0, 0, 1])
        else:
            p = in_p[0, 0, 0]
            exn = in_exn[0, 0, 0]

        # perform units conversion
        rho_gcm3 = 0.001 * in_rho[0, 0, 0]
        p_mbar = 0.01 * p[0, 0, 0]

        # compute the saturation mixing ratio of water vapor
        qvs = beta * in_ps[0, 0, 0] / (p[0, 0, 0] - in_ps[0, 0, 0])

        # compute the contribution of autoconversion to rain development
        ar = k1 * (in_qc[0, 0, 0] > a) * (in_qc[0, 0, 0] - a)

        # compute the contribution of accretion to rain development
        cr = k2 * in_qc[0, 0, 0] * (in_qr[0, 0, 0] ** 0.875)

        if rain_evaporation:
            # compute the contribution of evaporation to rain development
            c = 1.6 + 124.9 * ((rho_gcm3[0, 0, 0] * in_qr[0, 0, 0]) ** 0.2046)
            er = (
                (1.0 - in_qv[0, 0, 0] / qvs[0, 0, 0])
                * c[0, 0, 0]
                * ((rho_gcm3[0, 0, 0] * in_qr[0, 0, 0]) ** 0.525)
                / (
                    rho_gcm3[0, 0, 0]
                    * (5.4e5 + 2.55e6 / (p_mbar[0, 0, 0] * qvs[0, 0, 0]))
                )
            )

        # calculate the tendencies
        if not rain_evaporation:
            out_qc_tnd = -(ar[0, 0, 0] + cr[0, 0, 0])
            out_qr_tnd = ar[0, 0, 0] + cr[0, 0, 0]
        else:
            out_qv_tnd = er[0, 0, 0]
            out_qc_tnd = -(ar[0, 0, 0] + cr[0, 0, 0])
            out_qr_tnd = ar[0, 0, 0] + cr[0, 0, 0] - er[0, 0, 0]

        # compute the change over time in potential temperature
        if rain_evaporation:
            out_theta_tnd = -lhvw / exn[0, 0, 0] * er[0, 0, 0]


class KesslerSaturationAdjustment(DiagnosticComponent):
    """
    The saturation adjustment as predicted by the WRF implementation
    of the Kessler microphysics scheme.

    References
    ----------
    Doms, G., et al. (2015). A description of the nonhydrostatic regional \
        COSMO-model. Part II: Physical parameterization. \
        Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
    Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
        Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
        Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
    """

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
        "gas_constant_of_water_vapor": DataArray(461.52, attrs={"units": "J K^-1 kg^-1"}),
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        ),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        physical_constants=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None
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

        air_pressure_on_interface_levels : `bool`, optional
            :obj:`True` (respectively, :obj:`False`) if the input pressure
            field is defined at the interface (resp., main) levels.
            Defaults to :obj:`True`.
        backend : `obj`, optional
            TODO
        dtype : `numpy.dtype`, optional
            The data type for any :class:`numpy.ndarray` instantiated and
            used within this class.
        physical_constants : `dict`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'gas_constant_of_dry_air', in units compatible with \
                    [J K^-1 kg^-1];
                * 'gas_constant_of_water_vapor', in units compatible with \
                    [J K^-1 kg^-1];
                * 'latent_heat_of_vaporization_of_water', in units compatible with \
                    [J kg^-1];
                * 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
                    with [J K^-1 kg^-1].

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
        """
        # keep track of input arguments needed at run-time
        self._apoil = air_pressure_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type)

        # set physical parameters values
        pcs = get_physical_constants(self._d_physical_constants, physical_constants)

        # shortcuts
        rd = pcs["gas_constant_of_dry_air"]
        rv = pcs["gas_constant_of_water_vapor"]
        beta = rd / rv
        lhvw = pcs["latent_heat_of_vaporization_of_water"]
        cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz + 1))

        # allocate the gt4py storages collecting inputs and outputs
        self._in_ps = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._out_qv = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._out_qc = zeros(storage_shape, backend, dtype, default_origin=default_origin)

        # initialize the underlying gt4py stencil object
        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            rebuild=rebuild,
            module="kessler_saturation_adjustment",
            externals={
                "air_pressure_on_interface_levels": air_pressure_on_interface_levels,
                "beta": beta,
                "lhvw": lhvw,
                "cp": cp,
            },
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
        dims_on_interface_levels = (
            grid.x.dims[0],
            grid.y.dims[0],
            grid.z_on_interface_levels.dims[0],
        )

        return_dict = {
            "air_temperature": {"dims": dims, "units": "K"},
            "mass_fraction_of_water_vapor_in_air": {"dims": dims, "units": "g g^-1"},
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
        }

        if self._apoil:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "Pa",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}

        return return_dict

    @property
    def diagnostic_properties(self):
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "mass_fraction_of_water_vapor_in_air": {"dims": dims, "units": "g g^-1"},
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
        }

        return return_dict

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # extract the required model variables
        in_t = state["air_temperature"]
        in_qv = state["mass_fraction_of_water_vapor_in_air"]
        in_qc = state["mass_fraction_of_cloud_liquid_water_in_air"]
        if self._apoil:
            in_p = state["air_pressure_on_interface_levels"]
        else:
            in_p = state["air_pressure"]

        # compute the saturation water vapor pressure
        try:
            in_t.host_to_device()
            self._in_ps.data[...] = tetens_formula(in_t.data)
            self._in_ps._sync_state.state = self._in_ps.SyncState.SYNC_DEVICE_DIRTY
        except AttributeError:
            self._in_ps[...] = tetens_formula(in_t)

        # run the stencil
        self._stencil(
            in_p=in_p,
            in_ps=self._in_ps,
            in_t=in_t,
            in_qv=in_qv,
            in_qc=in_qc,
            out_qv=self._out_qv,
            out_qc=self._out_qc,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        # collect the diagnostics
        diagnostics = {mfwv: self._out_qv, mfcw: self._out_qc}

        return diagnostics

    @staticmethod
    def _stencil_defs(
        in_p: gt.storage.f64_sd,
        in_ps: gt.storage.f64_sd,
        in_t: gt.storage.f64_sd,
        in_qv: gt.storage.f64_sd,
        in_qc: gt.storage.f64_sd,
        out_qv: gt.storage.f64_sd,
        out_qc: gt.storage.f64_sd,
    ):
        # interpolate the pressure at the vertical main levels
        if air_pressure_on_interface_levels:
            p = 0.5 * (in_p[0, 0, 0] + in_p[0, 0, 1])
        else:
            p = in_p[0, 0, 0]

        # compute the saturation mixing ratio of water vapor
        qvs = beta * in_ps[0, 0, 0] / (p[0, 0, 0] - in_ps[0, 0, 0])

        # compute the amount of latent heat released by the condensation of cloud liquid water
        sat = (qvs[0, 0, 0] - in_qv[0, 0, 0]) / (
            1.0 + qvs[0, 0, 0] * 4093.0 * lhvw / (cp * (in_t[0, 0, 0] - 36) ** 2.0)
        )

        # compute the source term representing the evaporation of cloud liquid water
        dlt = (sat[0, 0, 0] <= in_qc[0, 0, 0]) * sat[0, 0, 0] + (
            sat[0, 0, 0] > in_qc[0, 0, 0]
        ) * in_qc[0, 0, 0]

        # perform the adjustment
        out_qv = in_qv[0, 0, 0] + dlt[0, 0, 0]
        out_qc = in_qc[0, 0, 0] - dlt[0, 0, 0]


class KesslerFallVelocity(DiagnosticComponent):
    """
    Calculate the raindrop fall velocity as prescribed by the Kessler
    microphysics scheme.

    References
    ----------
    Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
        Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
        Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
    """

    def __init__(
        self,
        domain,
        grid_type="numerical",
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None
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
        """
        super().__init__(domain, grid_type)

        self._exec_info = exec_info

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz))

        self._in_rho_s = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )
        self._out_vt = zeros(storage_shape, backend, dtype, default_origin=default_origin)

        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            rebuild=rebuild,
            module="kessler_fall_velocity",
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {"raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"}}

        return return_dict

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_rho = state["air_density"]
        in_qr = state[mfpw]
        self._in_rho_s[...] = in_rho[:, :, nz - 1 : nz]

        # try:
        #     in_rho.host_to_device()
        #     self._in_rho_s.data[...] = in_rho.data[:, :, nz - 1 : nz]
        #     self._in_rho_s._sync_state.state = self._in_rho_s.SyncState.SYNC_DEVICE_DIRTY
        # except AttributeError:
        #     self._in_rho_s[...] = in_rho.data[:, :, nz - 1 : nz]

        self._stencil(
            in_rho=in_rho,
            in_rho_s=self._in_rho_s,
            in_qr=in_qr,
            out_vt=self._out_vt,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        # collect the diagnostics
        diagnostics = {"raindrop_fall_velocity": self._out_vt}

        return diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_rho_s: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        out_vt: gt.storage.f64_sd,
    ):
        out_vt = (
            36.34
            * (1.0e-3 * in_rho[0, 0, 0] * (in_qr[0, 0, 0] > 0.0) * in_qr[0, 0, 0])
            ** 0.1346
            * (in_rho_s[0, 0, 0] / in_rho[0, 0, 0]) ** 0.5
        )


class KesslerSedimentation(ImplicitTendencyComponent):
    """
    Calculate the vertical derivative of the sedimentation flux for the mass
    fraction of precipitation water.
    """

    def __init__(
        self,
        domain,
        grid_type="numerical",
        sedimentation_flux_scheme="first_order_upwind",
        maximum_vertical_cfl=0.975,
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

        sedimentation_flux_scheme : `str`, optional
            The numerical sedimentation flux scheme. Please refer to
            :class:`~tasmania.SedimentationFlux` for the available options.
            Defaults to 'first_order_upwind'.
        maximum_vertical_cfl : `float`, optional
            Maximum allowed vertical CFL number. Defaults to 0.975.
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
        super().__init__(domain, grid_type, **kwargs)

        self._exec_info = exec_info

        sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
        self._nb = sflux.nb

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz + 1))
        self._out_qr = zeros(storage_shape, backend, dtype, default_origin=default_origin)
        self._out_dfdz = zeros(
            storage_shape, backend, dtype, default_origin=default_origin
        )

        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            rebuild=rebuild,
            module="kessler_sedimentation",
            externals={"sflux": sflux.__call__, "max_cfl": maximum_vertical_cfl},
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "height_on_interface_levels": {"dims": dims_z, "units": "m"},
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return {
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
        }

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state, timestep):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nbh = self.horizontal_boundary.nb
        nbv = self._nb

        dt = timestep.total_seconds()
        in_rho = state["air_density"]
        in_h = state["height_on_interface_levels"]
        in_qr = state[mfpw]
        in_vt = state["raindrop_fall_velocity"]

        self._stencil(
            in_rho=in_rho,
            in_h=in_h,
            in_qr=in_qr,
            in_vt=in_vt,
            out_dfdz=self._out_dfdz,
            out_qr=self._out_qr,
            dt=dt,
            origin={"_all_": (nbh, nbh, nbv)},
            domain=(nx - 2 * nbh, ny - 2 * nbh, nz - nbv),
            exec_info=self._exec_info,
        )

        # dh = self._in_h[:, :, :-1] - self._in_h[:, :, 1:]
        # x = np.where(
        #   self._in_vt > self._max_cfl * dh / timestep.total_seconds())
        # if x[0].size > 0:
        #   print('Number of gps violating vertical CFL: {:4d}'.format(x[0].size))

        tendencies = {mfpw: self._out_qr}
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_h: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        in_vt: gt.storage.f64_sd,
        out_dfdz: gt.storage.f64_sd,
        out_qr: gt.storage.f64_sd,
        *,
        dt: float
    ):
        # dh = in_h[0, 0, 0] - in_h[0, 0, 0]
        # tmp_vt = \
        # 	(in_vt[0, 0, 0] >  max_cfl * dh[0, 0, 0] / dt) * max_cfl * dh[0, 0, 0] / dt + \
        # 	(in_vt[0, 0, 0] <= max_cfl * dh[0, 0, 0] / dt) * in_vt[0, 0, 0]

        # out_dfdz = sflux(rho=in_rho, h=in_h, q=in_qr, vt=in_vt)

        # interpolate the geometric height at the model main levels
        tmp_h = 0.5 * (in_h[0, 0, 0] + in_h[0, 0, 1])

        # calculate the vertical derivative of the sedimentation flux
        out_dfdz = (
            in_rho[0, 0, -1] * in_qr[0, 0, -1] * in_vt[0, 0, -1]
            - in_rho[0, 0, 0] * in_qr[0, 0, 0] * in_vt[0, 0, 0]
        ) / (tmp_h[0, 0, -1] - tmp_h[0, 0, 0])

        out_qr = out_dfdz[0, 0, 0] / in_rho[0, 0, 0]
