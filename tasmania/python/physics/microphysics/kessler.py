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
from typing import Mapping, Optional, TYPE_CHECKING, Tuple

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    TendencyComponent,
)
from tasmania.python.physics.microphysics.utils import SedimentationFlux
from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import get_storage_shape, zeros
from tasmania.python.utils.meteo_utils import goff_gratch_formula, tetens_formula

if TYPE_CHECKING:
    from tasmania.python.grids.domain import Domain


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
        domain: "Domain",
        grid_type: str = "numerical",
        air_pressure_on_interface_levels: bool = True,
        tendency_of_air_potential_temperature_in_diagnostics: bool = False,
        rain_evaporation: bool = True,
        autoconversion_threshold: DataArray = _d_a,
        autoconversion_rate: DataArray = _d_k1,
        collection_rate: DataArray = _d_k2,
        saturation_vapor_pressure_formula: str = "tetens",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
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
        **kwargs
    ) -> None:
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
            `True` (respectively, `False`) if the input pressure
            field is defined at the interface (resp., main) levels.
            Defaults to `True`.
        tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
            `True` to include the tendency for the potential
            temperature in the output dictionary collecting the diagnostics,
            `False` otherwise. Defaults to `False`.
        rain_evaporation : `bool`, optional
            `True` if the evaporation of raindrops should be taken
            into account, `False` otherwise. Defaults to `True`.
        autoconversion_threshold : `sympl.DataArray`, optional
            Autoconversion threshold, in units compatible with [g g^-1].
        autoconversion_rate : `sympl.DataArray`, optional
            Autoconversion rate, in units compatible with [s^-1].
        collection_rate : `sympl.DataArray`, optional
            Rate of collection, in units compatible with [s^-1].
        saturation_vapor_pressure_formula : `str`, optional
            The formula giving the saturation water vapor pressure. Available options are:

                * 'tetens' (default) for the Tetens' equation;
                * 'goff_gratch' for the Goff-Gratch equation.

        physical_constants : `dict[str, sympl.DataArray]`, optional
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
            if saturation_vapor_pressure_formula == "goff_gratch"
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
        self._in_ps = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_qc_tnd = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_qr_tnd = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        if rain_evaporation:
            self._out_qv_tnd = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )
            self._out_theta_tnd = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin=default_origin,
                managed_memory=managed_memory,
            )

        # initialize the underlying gt4py stencil object
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            name=self.__class__.__name__,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            dtypes={"dtype": dtype},
            externals={
                "air_pressure_on_interface_levels": air_pressure_on_interface_levels,
                "beta": beta,
                "lhvw": pcs["latent_heat_of_vaporization_of_water"],
                "rain_evaporation": rain_evaporation,
            },
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
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
            mfwv: {"dims": dims, "units": "g g^-1"},
            mfcw: {"dims": dims, "units": "g g^-1"},
            mfpw: {"dims": dims, "units": "g g^-1"},
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
    def tendency_properties(self) -> taz_types.properties_dict_t:
        grid = self._grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            mfcw: {"dims": dims, "units": "g g^-1 s^-1"},
            mfpw: {"dims": dims, "units": "g g^-1 s^-1"},
        }

        if self._rain_evaporation:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}

            if not self._pttd:
                return_dict["air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        if self._rain_evaporation and self._pttd:
            grid = self._grid
            dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
            return {
                "tendency_of_air_potential_temperature": {"dims": dims, "units": "K s^-1"}
            }
        else:
            return {}

    def array_call(
        self, state: taz_types.gtstorage_dict_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
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
        tendencies = {mfcw: self._out_qc_tnd, mfpw: self._out_qr_tnd}
        if self._rain_evaporation:
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
        in_rho: gtscript.Field["dtype"],
        in_p: gtscript.Field["dtype"],
        in_ps: gtscript.Field["dtype"],
        in_exn: gtscript.Field["dtype"],
        in_qc: gtscript.Field["dtype"],
        in_qr: gtscript.Field["dtype"],
        out_qc_tnd: gtscript.Field["dtype"],
        out_qr_tnd: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"] = None,
        out_qv_tnd: gtscript.Field["dtype"] = None,
        out_theta_tnd: gtscript.Field["dtype"] = None,
        *,
        a: float,
        k1: float,
        k2: float
    ) -> None:
        from __externals__ import (
            air_pressure_on_interface_levels,
            beta,
            lhvw,
            rain_evaporation,
        )

        with computation(PARALLEL), interval(...):
            # interpolate the pressure and the Exner function at the vertical main levels
            if __INLINED(air_pressure_on_interface_levels):  # compile-time if
                p = 0.5 * (in_p[0, 0, 0] + in_p[0, 0, 1])
                exn = 0.5 * (in_exn[0, 0, 0] + in_exn[0, 0, 1])
            else:
                p = in_p
                exn = in_exn

            # compute the saturation mixing ratio of water vapor
            qvs = beta * in_ps / p

            # compute the contribution of autoconversion to rain development
            ar = k1 * (in_qc > a) * (in_qc - a)

            # compute the contribution of accretion to rain development
            cr = k2 * in_qc * (in_qr ** 0.875) if in_qr > 0 else 0

            if __INLINED(rain_evaporation):  # compile-time if
                # compute the contribution of evaporation to rain development
                er = (
                    0.0484794 * (qvs - in_qv) * (in_rho * in_qr) ** (13.0 / 20.0)
                    if in_qr > 0
                    else 0
                )

            # calculate the tendencies
            if __INLINED(not rain_evaporation):  # compile-time if
                out_qc_tnd = -(ar + cr)
                out_qr_tnd = ar + cr
            else:
                out_qv_tnd = er
                out_qc_tnd = -(ar + cr)
                out_qr_tnd = ar + cr - er

            # compute the change over time in potential temperature
            if __INLINED(rain_evaporation):  # compile-time if
                out_theta_tnd = -lhvw / exn * er


class KesslerSaturationAdjustmentDiagnostic(ImplicitTendencyComponent):
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
        domain: "Domain",
        grid_type: str = "numerical",
        air_pressure_on_interface_levels: bool = True,
        saturation_vapor_pressure_formula: str = "tetens",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
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
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        air_pressure_on_interface_levels : `bool`, optional
            `True` (respectively, `False`) if the input pressure
            field is defined at the interface (resp., main) levels.
            Defaults to `True`.
        saturation_vapor_pressure_formula : `str`, optional
            The formula giving the saturation water vapor pressure. Available options are:

                * 'tetens' (default) for the Tetens' equation;
                * 'goff_gratch' for the Goff-Gratch equation.

        physical_constants : `dict[str, sympl.DataArray]`, optional
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
        # keep track of input arguments needed at run-time
        self._apoil = air_pressure_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type)

        # set the formula calculating the saturation water vapor pressure
        self._swvf = (
            goff_gratch_formula
            if saturation_vapor_pressure_formula == "goff_gratch"
            else tetens_formula
        )

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
        self._in_ps = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_qv = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_qc = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_t = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_theta_tnd = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        # initialize the underlying gt4py stencil object
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            name=self.__class__.__name__,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            dtypes={"dtype": dtype},
            externals={
                "air_pressure_on_interface_levels": air_pressure_on_interface_levels,
                "beta": beta,
                "lhvw": lhvw,
                "cp": cp,
                "rv": rv,
            },
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
        dims_on_interface_levels = (
            grid.x.dims[0],
            grid.y.dims[0],
            grid.z_on_interface_levels.dims[0],
        )

        return_dict = {
            "air_temperature": {"dims": dims, "units": "K"},
            mfwv: {"dims": dims, "units": "g g^-1"},
            mfcw: {"dims": dims, "units": "g g^-1"},
        }

        if self._apoil:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "Pa",
            }
            return_dict["exner_function_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "J kg^-1 K^-1",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}
            return_dict["exner_function"] = {"dims": dims, "units": "J kg^-1 K^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {"air_potential_temperature": {"dims": dims, "units": "K s^-1"}}

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_temperature": {"dims": dims, "units": "K"},
            mfwv: {"dims": dims, "units": "g g^-1"},
            mfcw: {"dims": dims, "units": "g g^-1"},
        }

        return return_dict

    def array_call(
        self, state: taz_types.gtstorage_dict_t, timestep: taz_types.timedelta_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # extract the required model variables
        in_t = state["air_temperature"]
        in_qv = state[mfwv]
        in_qc = state[mfcw]
        if self._apoil:
            in_p = state["air_pressure_on_interface_levels"]
            in_exn = state["exner_function_on_interface_levels"]
        else:
            in_p = state["air_pressure"]
            in_exn = state["exner_function"]

        # compute the saturation water vapor pressure
        self._in_ps[...] = self._swvf(in_t)

        # run the stencil
        self._stencil(
            in_p=in_p,
            in_ps=self._in_ps,
            in_t=in_t,
            in_exn=in_exn,
            in_qv=in_qv,
            in_qc=in_qc,
            out_qv=self._out_qv,
            out_qc=self._out_qc,
            out_t=self._out_t,
            tnd_theta=self._out_theta_tnd,
            dt=timestep.total_seconds(),
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        # collect the tendencies and the diagnostics
        tendencies = {"air_potential_temperature": self._out_theta_tnd}
        diagnostics = {
            "air_temperature": self._out_t,
            mfwv: self._out_qv,
            mfcw: self._out_qc,
        }

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_p: gtscript.Field["dtype"],
        in_ps: gtscript.Field["dtype"],
        in_t: gtscript.Field["dtype"],
        in_exn: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"],
        in_qc: gtscript.Field["dtype"],
        out_qv: gtscript.Field["dtype"],
        out_qc: gtscript.Field["dtype"],
        out_t: gtscript.Field["dtype"],
        tnd_theta: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        from __externals__ import air_pressure_on_interface_levels, beta, cp, lhvw, rv

        with computation(PARALLEL), interval(...):
            # interpolate the pressure at the vertical main levels
            if __INLINED(air_pressure_on_interface_levels):  # compile-time if
                p = 0.5 * (in_p[0, 0, 0] + in_p[0, 0, 1])
                exn = 0.5 * (in_exn[0, 0, 0] + in_exn[0, 0, 1])
            else:
                p = in_p
                exn = in_exn

            # compute the saturation mixing ratio of water vapor
            qvs = beta * in_ps / p

            # compute the amount of latent heat released by the condensation of
            # cloud liquid water
            sat = (qvs - in_qv) / (1.0 + qvs * (lhvw ** 2.0) / (cp * rv * (in_t ** 2.0)))

            # compute the source term representing the evaporation of
            # cloud liquid water
            dq = sat if (sat <= in_qc) else in_qc

            # perform the adjustment
            out_qv = in_qv + dq
            out_qc = in_qc - dq
            out_t = in_t - dq * lhvw / cp

            # calculate the tendency of air potential temperature
            tnd_theta = (lhvw / exn) * (-dq / dt)


class KesslerSaturationAdjustmentPrognostic(TendencyComponent):
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
        domain: "Domain",
        grid_type: str = "numerical",
        air_pressure_on_interface_levels: bool = True,
        saturation_vapor_pressure_formula: str = "tetens",
        saturation_rate: Optional[DataArray] = None,
        physical_constants: Optional[Mapping[str, DataArray]] = None,
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
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        air_pressure_on_interface_levels : `bool`, optional
            `True` (respectively, `False`) if the input pressure
            field is defined at the interface (resp., main) levels.
            Defaults to `True`.
        saturation_vapor_pressure_formula : `str`, optional
            The formula giving the saturation water vapor pressure. Available options are:

                * 'tetens' (default) for the Tetens' equation;
                * 'goff_gratch' for the Goff-Gratch equation.

        saturation_rate : `sympl.DataArray`, optional
            Saturation rate, in units compatible with [s^-1].
        physical_constants : `dict[str, sympl.DataArray]`, optional
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
        # keep track of input arguments needed at run-time
        self._apoil = air_pressure_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, grid_type)

        # set the formula calculating the saturation water vapor pressure
        self._swvf = (
            goff_gratch_formula
            if saturation_vapor_pressure_formula == "goff_gratch"
            else tetens_formula
        )

        # get the saturation rate
        self._sr = (
            saturation_rate.to_units("s^-1").values.item()
            if saturation_rate is not None
            else 0.5
        )

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
        self._in_ps = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._tnd_qv = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._tnd_qc = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._tnd_theta = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        # initialize the underlying gt4py stencil object
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            name=self.__class__.__name__,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            dtypes={"dtype": dtype},
            externals={
                "air_pressure_on_interface_levels": air_pressure_on_interface_levels,
                "beta": beta,
                "lhvw": lhvw,
                "cp": cp,
                "rv": rv,
            },
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
        dims_on_interface_levels = (
            grid.x.dims[0],
            grid.y.dims[0],
            grid.z_on_interface_levels.dims[0],
        )

        return_dict = {
            "air_temperature": {"dims": dims, "units": "K"},
            mfwv: {"dims": dims, "units": "g g^-1"},
            mfcw: {"dims": dims, "units": "g g^-1"},
        }

        if self._apoil:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "Pa",
            }
            return_dict["exner_function_on_interface_levels"] = {
                "dims": dims_on_interface_levels,
                "units": "J kg^-1 K^-1",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}
            return_dict["exner_function"] = {"dims": dims, "units": "J kg^-1 K^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_potential_temperature": {"dims": dims, "units": "K s^-1"},
            mfwv: {"dims": dims, "units": "g g^-1 s^-1"},
            mfcw: {"dims": dims, "units": "g g^-1 s^-1"},
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.gtstorage_dict_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # extract the required model variables
        in_t = state["air_temperature"]
        in_qv = state[mfwv]
        in_qc = state[mfcw]
        if self._apoil:
            in_p = state["air_pressure_on_interface_levels"]
            in_exn = state["exner_function_on_interface_levels"]
        else:
            in_p = state["air_pressure"]
            in_exn = state["exner_function"]

        # compute the saturation water vapor pressure
        self._in_ps[...] = self._swvf(in_t)

        # run the stencil
        self._stencil(
            in_p=in_p,
            in_ps=self._in_ps,
            in_t=in_t,
            in_exn=in_exn,
            in_qv=in_qv,
            in_qc=in_qc,
            tnd_qv=self._tnd_qv,
            tnd_qc=self._tnd_qc,
            tnd_theta=self._tnd_theta,
            sr=self._sr,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        # collect the tendencies and the diagnostics
        tendencies = {
            "air_potential_temperature": self._tnd_theta,
            mfwv: self._tnd_qv,
            mfcw: self._tnd_qc,
        }
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_p: gtscript.Field["dtype"],
        in_ps: gtscript.Field["dtype"],
        in_t: gtscript.Field["dtype"],
        in_exn: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"],
        in_qc: gtscript.Field["dtype"],
        tnd_qv: gtscript.Field["dtype"],
        tnd_qc: gtscript.Field["dtype"],
        tnd_theta: gtscript.Field["dtype"],
        *,
        sr: float
    ) -> None:
        from __externals__ import air_pressure_on_interface_levels, beta, cp, lhvw, rv

        with computation(PARALLEL), interval(...):
            # interpolate the pressure at the vertical main levels
            if __INLINED(air_pressure_on_interface_levels):  # compile-time if
                p = 0.5 * (in_p[0, 0, 0] + in_p[0, 0, 1])
                exn = 0.5 * (in_exn[0, 0, 0] + in_exn[0, 0, 1])
            else:
                p = in_p
                exn = in_exn

            # compute the saturation mixing ratio of water vapor
            qvs = beta * in_ps / p

            # compute the amount of latent heat released by the condensation of
            # cloud liquid water
            sat = (qvs - in_qv) / (1.0 + qvs * (lhvw ** 2.0) / (cp * rv * (in_t ** 2.0)))

            # compute the source term representing the evaporation of
            # cloud liquid water
            dq = sat if (sat <= in_qc) else in_qc

            # calculate the tendencies
            tnd_qv = sr * dq
            tnd_qc = -sr * dq
            tnd_theta = -sr * (lhvw / exn) * dq


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
        domain: "Domain",
        grid_type: str = "numerical",
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
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

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
        super().__init__(domain, grid_type)

        self._exec_info = exec_info

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz))

        self._in_rho_s = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_vt = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            name=self.__class__.__name__,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            dtypes={"dtype": dtype},
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            mfpw: {"dims": dims, "units": "g g^-1"},
        }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {"raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"}}

        return return_dict

    def array_call(self, state: taz_types.gtstorage_dict_t) -> taz_types.gtstorage_dict_t:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        in_rho = state["air_density"]
        in_qr = state[mfpw]
        self._in_rho_s[...] = in_rho[:, :, nz - 1 : nz]

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
        in_rho: gtscript.Field["dtype"],
        in_rho_s: gtscript.Field["dtype"],
        in_qr: gtscript.Field["dtype"],
        out_vt: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_vt = (
                36.34
                * (1.0e-3 * in_rho * (in_qr > 0.0) * in_qr) ** 0.1346
                * (in_rho_s / in_rho) ** 0.5
            )


class KesslerSedimentation(ImplicitTendencyComponent):
    """
    Calculate the vertical derivative of the sedimentation flux for the mass
    fraction of precipitation water.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str = "numerical",
        sedimentation_flux_scheme: str = "first_order_upwind",
        maximum_vertical_cfl: float = 0.975,
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
        **kwargs
    ) -> None:
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
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent class.
        """
        super().__init__(domain, grid_type, **kwargs)

        self._exec_info = exec_info

        sflux = SedimentationFlux.factory(sedimentation_flux_scheme)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = get_storage_shape(storage_shape, (nx, ny, nz + 1))
        self._out_qr = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        self._out_dfdz = zeros(
            storage_shape,
            backend,
            dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            name=self.__class__.__name__,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            dtypes={"dtype": dtype},
            externals={
                "sflux": sflux.__call__,
                "sflux_extent": sflux.nb,
                # "max_cfl": maximum_vertical_cfl},
            },
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "height_on_interface_levels": {"dims": dims_z, "units": "m"},
            mfpw: {"dims": dims, "units": "g g^-1"},
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return {mfpw: {"dims": dims, "units": "g g^-1 s^-1"}}

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.gtstorage_dict_t, timestep: taz_types.timedelta_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nbh = 0  # self.horizontal_boundary.nb if self.grid_type == "numerical" else 0

        in_rho = state["air_density"]
        in_h = state["height_on_interface_levels"]
        in_qr = state[mfpw]
        in_vt = state["raindrop_fall_velocity"]

        dh = in_h[:, :, :-1] - in_h[:, :, 1:]
        idx = np.where(
            in_vt[:-1, :-1, :-1] > 0.975 * dh[:-1, :-1] / timestep.total_seconds()
        )
        if idx[0].size > 0:
            print("Number of gps violating vertical CFL: {:4d}".format(idx[0].size))
        in_vt[idx] = 0.975 * dh[idx]

        self._stencil(
            in_rho=in_rho,
            in_h=in_h,
            in_qr=in_qr,
            in_vt=in_vt,
            out_qr=self._out_qr,
            origin={"_all_": (nbh, nbh, 0)},
            domain=(nx - 2 * nbh, ny - 2 * nbh, nz),
            exec_info=self._exec_info,
        )

        tendencies = {mfpw: self._out_qr}
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gtscript.Field["dtype"],
        in_h: gtscript.Field["dtype"],
        in_qr: gtscript.Field["dtype"],
        in_vt: gtscript.Field["dtype"],
        out_qr: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import sflux, sflux_extent

        with computation(PARALLEL), interval(0, None):
            h = 0.5 * (in_h[0, 0, 0] + in_h[0, 0, 1])

        with computation(FORWARD), interval(0, sflux_extent):
            out_qr = 0.0
        with computation(PARALLEL), interval(sflux_extent, None):
            out_dfdz = sflux(rho=in_rho, h=h, q=in_qr, vt=in_vt)
            out_qr = out_dfdz / in_rho
