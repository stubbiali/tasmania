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

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@gtscript.function
def first_order_boundary(dz, w, phi):
    out = (w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 0] * phi[0, 0, 0]) / dz
    return out


@gtscript.function
def second_order_boundary(dz, w, phi):
    out = (
        0.5
        * (
            -3.0 * w[0, 0, 0] * phi[0, 0, 0]
            + 4.0 * w[0, 0, -1] * phi[0, 0, -1]
            - w[0, 0, -2] * phi[0, 0, -2]
        )
        / dz
    )
    return out


class IsentropicVerticalAdvection(TendencyComponent):
    """
    This class inherits :class:`tasmania.TendencyComponent` to calculate
    the vertical derivative of the conservative vertical advection flux
    in isentropic coordinates for any prognostic variable included in
    the isentropic model. The class is always instantiated over the
    numerical grid of the underlying domain.
    """

    def __init__(
        self,
        domain,
        flux_scheme="upwind",
        moist=False,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        managed_memory=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        flux_scheme : `str`, optional
            The numerical flux scheme to implement. Defaults to 'upwind'.
            See :class:`~tasmania.IsentropicMinimalVerticalFlux` for all
            available options.
        moist : `bool`, optional
            `True` if water species are included in the model,
            `False` otherwise. Defaults to `False`.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            `True` if the input tendency of air potential temperature
            is defined at the interface levels, `False` otherwise.
            Defaults to `False`.
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
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, "numerical", **kwargs)

        # instantiate the object calculating the flux
        self._vflux = IsentropicMinimalVerticalFlux.factory(flux_scheme)

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = (nx, ny, nz + 1) if storage_shape is None else storage_shape
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz + 1)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz + 1, error_msg

        # allocate the gt4py storages collecting the stencil outputs
        self._out_s = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._out_su = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._out_sv = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        if moist:
            self._out_qv = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._out_qc = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._out_qr = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )

        # instantiate the underlying stencil object
        externals = {
            "compute_boundary": first_order_boundary
            if self._vflux.order == 1
            else second_order_boundary,
            "moist": moist,
            "vflux": self._vflux.__call__,
            "vflux_end": -self._vflux.extent + 1 if self._vflux.extent > 1 else None,
            "vflux_extent": self._vflux.extent,
            "vstaggering": self._stgz,
        }
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            backend=backend,
            build_info=build_info,
            externals=externals,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    @property
    def input_properties(self):
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        if self._stgz:
            dims_stgz = (
                grid.x.dims[0],
                grid.y.dims[0],
                grid.z_on_interface_levels.dims[0],
            )
            return_dict["tendency_of_air_potential_temperature_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "K s^-1",
            }
        else:
            return_dict["tendency_of_air_potential_temperature"] = {
                "dims": dims,
                "units": "K s^-1",
            }

        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }

        return return_dict

    @property
    def tendency_properties(self):
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }
        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }

        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dz = self.grid.dz.to_units("K").values.item()
        nb = self._vflux.extent

        # grab the required model variables
        in_w = (
            state["tendency_of_air_potential_temperature_on_interface_levels"]
            if self._stgz
            else state["tendency_of_air_potential_temperature"]
        )
        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]
        if self._moist:
            in_qv = state[mfwv]
            in_qc = state[mfcw]
            in_qr = state[mfpw]

        # set the stencil's arguments
        stencil_args = {
            "dz": dz,
            "in_w": in_w,
            "in_s": in_s,
            "out_s": self._out_s,
            "in_su": in_su,
            "out_su": self._out_su,
            "in_sv": in_sv,
            "out_sv": self._out_sv,
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": in_qv,
                    "out_qv": self._out_qv,
                    "in_qc": in_qc,
                    "out_qc": self._out_qc,
                    "in_qr": in_qr,
                    "out_qr": self._out_qr,
                }
            )

        # run the stencil
        self._stencil(
            **stencil_args,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info
        )

        # collect the output arrays in a dictionary
        tendencies = {
            "air_isentropic_density": self._out_s,
            "x_momentum_isentropic": self._out_su,
            "y_momentum_isentropic": self._out_sv,
        }
        if self._moist:
            tendencies[mfwv] = self._out_qv
            tendencies[mfcw] = self._out_qc
            tendencies[mfpw] = self._out_qr

        return tendencies, {}

    @staticmethod
    def _stencil_defs(
        in_w: gtscript.Field[np.float64],
        in_s: gtscript.Field[np.float64],
        in_su: gtscript.Field[np.float64],
        in_sv: gtscript.Field[np.float64],
        out_s: gtscript.Field[np.float64],
        out_su: gtscript.Field[np.float64],
        out_sv: gtscript.Field[np.float64],
        in_qv: gtscript.Field[np.float64] = None,
        in_qc: gtscript.Field[np.float64] = None,
        in_qr: gtscript.Field[np.float64] = None,
        out_qv: gtscript.Field[np.float64] = None,
        out_qc: gtscript.Field[np.float64] = None,
        out_qr: gtscript.Field[np.float64] = None,
        *,
        dt: float = 0.0,
        dz: float
    ):
        from __externals__ import (
            compute_boundary,
            moist,
            vflux,
            vflux_end,
            vflux_extent,
            vstaggering,
        )

        # interpolate the velocity on the interface levels
        with computation(PARALLEL):
            with interval(0, 1):
                w = 0.0
            with interval(1, None):
                if vstaggering:  # compile-time if
                    w = in_w
                else:
                    w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, -1])

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if vstaggering:
                wc = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                wc = in_w

        # compute the isentropic density of the water species
        with computation(PARALLEL), interval(0, None):
            if moist:  # compile-time if
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr
            else:
                sqv = 0.0  # dummy computation

        # compute the fluxes
        with computation(PARALLEL), interval(vflux_extent, vflux_end):
            if not moist:  # compile-time if
                flux_s, flux_su, flux_sv = vflux(
                    dt=dt, dz=dz, w=w, s=in_s, su=in_su, sv=in_sv
                )
            else:
                flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = vflux(
                    dt=dt,
                    dz=dz,
                    w=w,
                    s=in_s,
                    su=in_su,
                    sv=in_sv,
                    sqv=sqv,
                    sqc=sqc,
                    sqr=sqr,
                )

        # calculate the tendencies
        with computation(PARALLEL):
            with interval(0, vflux_extent):
                out_s = 0.0
                out_su = 0.0
                out_sv = 0.0
                if moist:  # compile-time if
                    out_qv = 0.0
                    out_qc = 0.0
                    out_qr = 0.0

            with interval(vflux_extent, -vflux_extent):
                out_s = (flux_s[0, 0, 1] - flux_s[0, 0, 0]) / dz
                out_su = (flux_su[0, 0, 1] - flux_su[0, 0, 0]) / dz
                out_sv = (flux_sv[0, 0, 1] - flux_sv[0, 0, 0]) / dz
                if moist:  # compile-time if
                    out_qv = (flux_sqv[0, 0, 1] - flux_sqv[0, 0, 0]) / (
                        in_s[0, 0, 0] * dz
                    )
                    out_qc = (flux_sqc[0, 0, 1] - flux_sqc[0, 0, 0]) / (
                        in_s[0, 0, 0] * dz
                    )

                    out_qr = (flux_sqr[0, 0, 1] - flux_sqr[0, 0, 0]) / (
                        in_s[0, 0, 0] * dz
                    )

            with interval(-vflux_extent, None):
                out_s = compute_boundary(dz=dz, w=wc, phi=in_s)
                out_su = compute_boundary(dz=dz, w=wc, phi=in_su)
                out_sv = compute_boundary(dz=dz, w=wc, phi=in_sv)
                if moist:  # compile-time if
                    tmp_qv = compute_boundary(dz=dz, w=wc, phi=sqv)
                    out_qv = tmp_qv / in_s
                    tmp_qc = compute_boundary(dz=dz, w=wc, phi=sqc)
                    out_qc = tmp_qc / in_s
                    tmp_qr = compute_boundary(dz=dz, w=wc, phi=sqr)
                    out_qr = tmp_qr / in_s


class PrescribedSurfaceHeating(TendencyComponent):
    """
    Calculate the variation in air potential temperature as prescribed
    in the reference paper, namely
        .. math::
            \dot{\theta} =
            \Biggl \lbrace
            {
                \\frac{\theta \, R_d \, \alpha(t)}{p \, C_p}
                \exp[\left( - \alpha(t) \left( z - h_s \\right) \\right]}
                \left[ F_0^{sw}(t) \sin{\left( \omega^{sw} (t - t_0) \\right)}
                + F_0^{fw}(t) \sin{\left( \omega^{fw} (t - t_0) \\right)} \\right]
                \text{if} {r = \sqrt{x^2 + y^2} < R}
                \atop
                0 \text{otherwise}
            } .
    The class is always instantiated over the numerical grid of the
    underlying domain.
    References
    ----------
    Reisner, J. M., and P. K. Smolarkiewicz. (1994). \
        Thermally forced low Froude number flow past three-dimensional obstacles. \
        *Journal of Atmospheric Sciences*, *51*(1):117-133.
    """

    # Default values for the physical constants used in the class
    _d_physical_constants = {
        "gas_constant_of_dry_air": DataArray(287.0, attrs={"units": "J K^-1 kg^-1"}),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=None,
        amplitude_at_day_fw=None,
        amplitude_at_night_sw=None,
        amplitude_at_night_fw=None,
        frequency_sw=None,
        frequency_fw=None,
        attenuation_coefficient_at_day=None,
        attenuation_coefficient_at_night=None,
        characteristic_length=None,
        starting_time=None,
        backend="numpy",
        physical_constants=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
            :obj:`True` to place the calculated tendency of air
            potential temperature in the ``diagnostics`` output
            dictionary, :obj:`False` to regularly place it in the
            `tendencies` dictionary. Defaults to :obj:`False`.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            :obj:`True` (respectively, :obj:`False`) if the tendency
            of air potential temperature should be calculated at the
            interface (resp., main) vertical levels. Defaults to :obj:`False`.
        air_pressure_on_interface_levels : `bool`, optional
            :obj:`True` (respectively, :obj:`False`) if the input
            air potential pressure is defined at the interface
            (resp., main) vertical levels. Defaults to :obj:`True`.
        amplitude_at_day_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at day,
            in units compatible with [W m^-2].
        amplitude_at_day_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at day,
            in units compatible with [W m^-2].
        amplitude_at_night_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at night,
            in units compatible with [W m^-2].
        amplitude_at_night_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at night,
            in units compatible with [W m^-2].
        frequency_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\omega^{sw}`,
            in units compatible with [s^-1].
        frequency_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\omega^{fw}`,
            in units compatible with [s^-1].
        attenuation_coefficient_at_day : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\alpha` at day,
            in units compatible with [m^-1].
        attenuation_coefficient_at_night : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\alpha` at night,
            in units compatible with [m^-1].
        characteristic_length : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`R`,
            in units compatible with [m].
        starting_time : `datetime`, optional
            The time :math:`t_0` when surface heating/cooling is triggered.
        backend : `obj`, optional
            TODO
        physical_constants : `dict`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`~sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:
                * 'gas_constant_of_dry_air', in units compatible with \
                    [J K^-1 kg^-1];
                * 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
                    with [J K^-1 kg^-1].
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`sympl.TendencyComponent`.
        """
        self._tid = tendency_of_air_potential_temperature_in_diagnostics
        self._apil = air_pressure_on_interface_levels
        self._aptil = (
            tendency_of_air_potential_temperature_on_interface_levels
            and air_pressure_on_interface_levels
        )
        self._backend = backend

        super().__init__(domain, "numerical", **kwargs)

        self._f0d_sw = (
            amplitude_at_day_sw.to_units("W m^-2").values.item()
            if amplitude_at_day_sw is not None
            else 800.0
        )
        self._f0d_fw = (
            amplitude_at_day_fw.to_units("W m^-2").values.item()
            if amplitude_at_day_fw is not None
            else 400.0
        )
        self._f0n_sw = (
            amplitude_at_night_sw.to_units("W m^-2").values.item()
            if amplitude_at_night_sw is not None
            else -75.0
        )
        self._f0n_fw = (
            amplitude_at_night_fw.to_units("W m^-2").values.item()
            if amplitude_at_night_fw is not None
            else -37.5
        )
        self._w_sw = (
            frequency_sw.to_units("hr^-1").values.item()
            if frequency_sw is not None
            else np.pi / 12.0
        )
        self._w_fw = (
            frequency_fw.to_units("hr^-1").values.item()
            if frequency_fw is not None
            else np.pi
        )
        self._ad = (
            attenuation_coefficient_at_day.to_units("m^-1").values.item()
            if attenuation_coefficient_at_day is not None
            else 1.0 / 600.0
        )
        self._an = (
            attenuation_coefficient_at_night.to_units("m^-1").values.item()
            if attenuation_coefficient_at_night is not None
            else 1.0 / 75.0
        )
        self._cl = (
            characteristic_length.to_units("m").values.item()
            if characteristic_length is not None
            else 25000.0
        )
        self._t0 = starting_time

        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._rd = pcs["gas_constant_of_dry_air"]
        self._cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {"height_on_interface_levels": {"dims": dims_stgz, "units": "m"}}

        if self._apil:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "Pa",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid

        return_dict = {}

        if not self._tid:
            if self._aptil:
                dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
                return_dict["air_potential_temperature_on_interface_levels"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }
            else:
                dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
                return_dict["air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    @property
    def diagnostic_properties(self):
        g = self.grid

        return_dict = {}

        if self._tid:
            if self._aptil:
                dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
                return_dict[
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ] = {"dims": dims, "units": "K s^-1"}
            else:
                dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
                return_dict["tendency_of_air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    def array_call(self, state):
        g = self.grid
        mi, mj = g.nx, g.ny
        mk = g.nz + 1 if self._aptil else g.nz

        t = state["time"]
        dt = (t - self._t0).total_seconds() / 3600.0 if self._t0 is not None else t.hour

        if dt <= 0.0:
            out = np.zeros((mi, mj, mk), dtype=state["height_on_interface_levels"].dtype)
        else:
            x = g.x.to_units("m").values[:, np.newaxis, np.newaxis]
            y = g.y.to_units("m").values[np.newaxis, :, np.newaxis]
            theta1d = (
                g.z_on_interface_levels.to_units("K").values
                if self._aptil
                else g.z.to_units("K").values
            )
            theta = theta1d[np.newaxis, np.newaxis, :]

            pv = (
                state["air_pressure_on_interface_levels"]
                if self._apil
                else state["air_pressure"]
            )
            p = pv if pv.shape[2] == mk else 0.5 * (pv[:, :, 1:] + pv[:, :, :-1])
            zv = state["height_on_interface_levels"]
            z = zv if self._aptil else 0.5 * (zv[:, :, 1:] + zv[:, :, :-1])
            h = zv[:, :, -1:]

            w_sw = self._w_sw
            w_fw = self._w_fw
            cl = self._cl

            t_in_seconds = t.hour * 60 * 60 + t.minute * 60 + t.second
            t_sw = (2 * np.pi / w_sw) * 60 * 60
            day = int(t_in_seconds / t_sw) % 2 == 0
            f0_sw = self._f0d_sw if day else self._f0n_sw
            f0_fw = self._f0d_fw if day else self._f0n_fw
            a = self._ad if day else self._an

            out = (
                theta
                * self._rd
                * a
                / (p * self._cp)
                * np.exp(-a * (z - h))
                * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
                * (x ** 2 + y ** 2 < cl ** 2)
            )

        tendencies = {}
        if not self._tid:
            if self._aptil:
                tendencies["air_potential_temperature_on_interface_levels"] = out
            else:
                tendencies["air_potential_temperature"] = out

        diagnostics = {}
        if self._tid:
            if self._aptil:
                diagnostics[
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ] = out
            else:
                diagnostics["tendency_of_air_potential_temperature"] = out

        return tendencies, diagnostics



