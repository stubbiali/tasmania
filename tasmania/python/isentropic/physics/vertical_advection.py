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
from typing import Dict, Optional, Sequence, TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.core_components import TendencyComponent
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.data import get_physical_constants

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict, PropertyDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@gtscript.function
def first_order_boundary(
    dz: float, w: ty.GTField, phi: ty.GTField
) -> ty.GTField:
    out = (w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 0] * phi[0, 0, 0]) / dz
    return out


@gtscript.function
def second_order_boundary(
    dz: float, w: ty.GTField, phi: ty.GTField
) -> ty.GTField:
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
        domain: "Domain",
        grid_type: str = "numerical",
        flux_scheme: str = "upwind",
        moist: bool = False,
        tendency_of_air_potential_temperature_on_interface_levels: bool = False,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None,
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
        flux_scheme : `str`, optional
            The numerical flux scheme to implement. Defaults to 'upwind'.
            See :class:`~tasmania.IsentropicMinimalVerticalFlux` for all
            available options.
        moist : `bool`, optional
            ``True`` if water species are included in the model,
            ``False`` otherwise. Defaults to ``False``.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            ``True`` if the input tendency of air potential temperature
            is defined at the interface levels, ``False`` otherwise.
            Defaults to ``False``.
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
            class.
        """
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels

        # call parent's constructor
        super().__init__(
            domain,
            grid_type,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
            **kwargs
        )

        # instantiate the object calculating the flux
        self._vflux = IsentropicMinimalVerticalFlux.factory(
            flux_scheme, backend=self.backend
        )

        # instantiate the underlying stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            # "compute_boundary": self.get_subroutine_definition("first_order_boundary")
            # if self._vflux.order == 1
            # else self.get_subroutine_definition("second_order_boundary"),
            "flux_end": -self._vflux.extent + 1
            if self._vflux.extent > 1
            else None,
            "flux_extent": self._vflux.extent,
            "get_flux_dry": self._vflux.get_subroutine_definition("flux_dry"),
            "get_flux_moist": self._vflux.get_subroutine_definition(
                "flux_moist"
            ),
            "moist": moist,
            "set_output": self.get_subroutine_definition("set_output"),
            "staggering": self._stgz,
        }
        self._stencil = self.compile_stencil("stencil")

    @property
    def input_properties(self) -> "PropertyDict":
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

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

        if self._stgz:
            dims_stgz = (
                grid.x.dims[0],
                grid.y.dims[0],
                grid.z_on_interface_levels.dims[0],
            )
            return_dict[
                "tendency_of_air_potential_temperature_on_interface_levels"
            ] = {
                "dims": dims_stgz,
                "units": "K s^-1",
            }
        else:
            return_dict["tendency_of_air_potential_temperature"] = {
                "dims": dims,
                "units": "K s^-1",
            }

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> "PropertyDict":
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

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
    def diagnostic_properties(self) -> "PropertyDict":
        return {}

    def array_call(
        self,
        state: "NDArrayLikeDict",
        out_tendencies: "NDArrayLikeDict",
        out_diagnostics: "NDArrayLikeDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        # shortcuts
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dz = self.grid.dz.to_units("K").values.item()

        # set the stencil's arguments
        stencil_args = {
            "dz": dz,
            "in_w": state[
                "tendency_of_air_potential_temperature_on_interface_levels"
            ]
            if self._stgz
            else state["tendency_of_air_potential_temperature"],
            "in_s": state["air_isentropic_density"],
            "out_s": out_tendencies["air_isentropic_density"],
            "ow_out_s": overwrite_tendencies["air_isentropic_density"],
            "in_su": state["x_momentum_isentropic"],
            "out_su": out_tendencies["x_momentum_isentropic"],
            "ow_out_su": overwrite_tendencies["x_momentum_isentropic"],
            "in_sv": state["y_momentum_isentropic"],
            "out_sv": out_tendencies["y_momentum_isentropic"],
            "ow_out_sv": overwrite_tendencies["y_momentum_isentropic"],
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": state[mfwv],
                    "out_qv": out_tendencies[mfwv],
                    "ow_out_qv": overwrite_tendencies[mfwv],
                    "in_qc": state[mfcw],
                    "out_qc": out_tendencies[mfcw],
                    "ow_out_qc": overwrite_tendencies[mfcw],
                    "in_qr": state[mfpw],
                    "out_qr": out_tendencies[mfpw],
                    "ow_out_qr": overwrite_tendencies[mfpw],
                }
            )

        # run the stencil
        self._stencil(
            **stencil_args,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args
        )

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil")
    def _stencil_numpy(
        self,
        in_w: np.ndarray,
        in_s: np.ndarray,
        in_su: np.ndarray,
        in_sv: np.ndarray,
        out_s: np.ndarray,
        out_su: np.ndarray,
        out_sv: np.ndarray,
        in_qv: Optional[np.ndarray] = None,
        in_qc: Optional[np.ndarray] = None,
        in_qr: Optional[np.ndarray] = None,
        out_qv: Optional[np.ndarray] = None,
        out_qc: Optional[np.ndarray] = None,
        out_qr: Optional[np.ndarray] = None,
        *,
        dt: float = 0.0,
        dz: float,
        ow_out_s: bool,
        ow_out_su: bool,
        ow_out_sv: bool,
        ow_out_qv: bool = True,
        ow_out_qc: bool = True,
        ow_out_qr: bool = True,
        origin: "TripletInt",
        domain: "TripletInt"
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        nb = self._vflux.extent

        # interpolate the velocity on the interface levels
        if self._stgz:
            w = in_w
        else:
            w = np.zeros_like(in_w)
            w[i, j, kstart + 1 : kstop] = 0.5 * (
                in_w[i, j, kstart + 1 : kstop] + in_w[i, j, kstart : kstop - 1]
            )

        # compute the isentropic density of the water species
        if self._moist:
            sqv = np.zeros_like(in_qv)
            sqv[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qv[i, j, kstart:kstop]
            )
            sqc = np.zeros_like(in_qc)
            sqc[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qc[i, j, kstart:kstop]
            )
            sqr = np.zeros_like(in_qr)
            sqr[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qr[i, j, kstart:kstop]
            )
        else:
            sqv = sqc = sqr = None

        # compute the fluxes
        fs, fsu, fsv = self._vflux.get_subroutine_definition("flux_dry")(
            dt=dt, dz=dz, w=w, s=in_s, su=in_su, sv=in_sv
        )

        # calculate the tendency for the air isentropic density
        tmp_out_s = np.zeros_like(in_s)
        tmp_out_s[i, j, kstart + nb : kstop - nb] = (
            fs[i, j, 1:] - fs[i, j, :-1]
        ) / dz
        set_output(out_s, tmp_out_s, ow_out_s)

        # calculate the tendency for the x-momentum
        tmp_out_su = np.zeros_like(in_s)
        tmp_out_su[i, j, kstart + nb : kstop - nb] = (
            fsu[i, j, 1:] - fsu[i, j, :-1]
        ) / dz
        set_output(out_su, tmp_out_su, ow_out_su)

        # calculate the tendency for the y-momentum
        tmp_out_sv = np.zeros_like(in_s)
        tmp_out_sv[i, j, kstart + nb : kstop - nb] = (
            fsv[i, j, 1:] - fsv[i, j, :-1]
        ) / dz
        set_output(out_sv, tmp_out_sv, ow_out_sv)

        if self._moist:
            # compute the fluxes
            fsqv, fsqc, fsqr = self._vflux.get_subroutine_definition(
                "flux_moist"
            )(dt=dt, dz=dz, w=w, sqv=sqv, sqc=sqc, sqr=sqr)

            # calculate the tendency for the water vapor
            tmp_out_qv = np.zeros_like(in_qv)
            tmp_out_qv[i, j, kstart + nb : kstop - nb] = (
                fsqv[i, j, 1:] - fsqv[i, j, :-1]
            ) / (in_s[i, j, kstart + nb : kstop - nb] * dz)
            set_output(out_qv, tmp_out_qv, ow_out_qv)

            # calculate the tendency for the cloud liquid water
            tmp_out_qc = np.zeros_like(in_qc)
            tmp_out_qc[i, j, kstart + nb : kstop - nb] = (
                fsqc[i, j, 1:] - fsqc[i, j, :-1]
            ) / (in_s[i, j, kstart + nb : kstop - nb] * dz)
            set_output(out_qc, tmp_out_qc, ow_out_qc)

            # calculate the tendency for the precipitation water
            tmp_out_qr = np.zeros_like(in_qr)
            tmp_out_qr[i, j, kstart + nb : kstop - nb] = (
                fsqr[i, j, 1:] - fsqr[i, j, :-1]
            ) / (in_s[i, j, kstart + nb : kstop - nb] * dz)
            set_output(out_qr, tmp_out_qr, ow_out_qr)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil")
    def _stencil_gt4py(
        in_w: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        out_s: gtscript.Field["dtype"],
        out_su: gtscript.Field["dtype"],
        out_sv: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"] = None,
        in_qc: gtscript.Field["dtype"] = None,
        in_qr: gtscript.Field["dtype"] = None,
        out_qv: gtscript.Field["dtype"] = None,
        out_qc: gtscript.Field["dtype"] = None,
        out_qr: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dz: float,
        ow_out_s: bool,
        ow_out_su: bool,
        ow_out_sv: bool,
        ow_out_qv: bool = True,
        ow_out_qc: bool = True,
        ow_out_qr: bool = True,
    ) -> None:
        from __externals__ import (
            flux_end,
            flux_extent,
            get_flux_dry,
            get_flux_moist,
            moist,
            set_output,
            staggering,
        )

        # interpolate the velocity on the interface levels
        with computation(FORWARD), interval(0, 1):
            w = 0.0
        with computation(PARALLEL), interval(1, None):
            if __INLINED(staggering):  # compile-time if
                w = in_w
            else:
                w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, -1])

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if __INLINED(staggering):
                wc = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                wc = in_w

        # compute the isentropic density of the water species
        if __INLINED(moist):  # compile-time if
            with computation(PARALLEL), interval(0, None):
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr

        # compute the fluxes
        with computation(PARALLEL), interval(flux_extent, flux_end):
            flux_s, flux_su, flux_sv = get_flux_dry(
                dt=dt, dz=dz, w=w, s=in_s, su=in_su, sv=in_sv
            )
            if __INLINED(moist):  # compile-time if
                flux_sqv, flux_sqc, flux_sqr = get_flux_moist(
                    dt=dt, dz=dz, w=w, sqv=sqv, sqc=sqc, sqr=sqr
                )

        # calculate the tendencies
        with computation(PARALLEL), interval(0, flux_extent):
            out_s = set_output(out_s, 0.0, ow_out_s)
            out_su = set_output(out_su, 0.0, ow_out_su)
            out_sv = set_output(out_sv, 0.0, ow_out_sv)
            if __INLINED(moist):  # compile-time if
                out_qv = set_output(out_qv, 0.0, ow_out_qv)
                out_qc = set_output(out_qc, 0.0, ow_out_qc)
                out_qr = set_output(out_qr, 0.0, ow_out_qr)
        with computation(PARALLEL), interval(flux_extent, -flux_extent):
            tmp_out_s = (flux_s[0, 0, 1] - flux_s[0, 0, 0]) / dz
            out_s = set_output(out_s, tmp_out_s, ow_out_s)
            tmp_out_su = (flux_su[0, 0, 1] - flux_su[0, 0, 0]) / dz
            out_su = set_output(out_su, tmp_out_su, ow_out_su)
            tmp_out_sv = (flux_sv[0, 0, 1] - flux_sv[0, 0, 0]) / dz
            out_sv = set_output(out_sv, tmp_out_sv, ow_out_sv)
            if __INLINED(moist):  # compile-time if
                tmp_out_qv = (flux_sqv[0, 0, 1] - flux_sqv[0, 0, 0]) / (
                    in_s[0, 0, 0] * dz
                )
                out_qv = set_output(out_qv, tmp_out_qv, ow_out_qv)
                tmp_out_qc = (flux_sqc[0, 0, 1] - flux_sqc[0, 0, 0]) / (
                    in_s[0, 0, 0] * dz
                )
                out_qc = set_output(out_qc, tmp_out_qc, ow_out_qc)
                tmp_out_qr = (flux_sqr[0, 0, 1] - flux_sqr[0, 0, 0]) / (
                    in_s[0, 0, 0] * dz
                )
                out_qr = set_output(out_qr, tmp_out_qr, ow_out_qr)
        with computation(PARALLEL), interval(-flux_extent, None):
            out_s = set_output(out_s, 0.0, ow_out_s)
            out_su = set_output(out_su, 0.0, ow_out_su)
            out_sv = set_output(out_sv, 0.0, ow_out_sv)
            if __INLINED(moist):  # compile-time if
                out_qv = set_output(out_qv, 0.0, ow_out_qv)
                out_qc = set_output(out_qc, 0.0, ow_out_qc)
                out_qr = set_output(out_qr, 0.0, ow_out_qr)


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
        "gas_constant_of_dry_air": DataArray(
            287.0, attrs={"units": "J K^-1 kg^-1"}
        ),
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
            The :class:`~tasmania.Domain` holding the grid underneath.
        tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
            ``True`` to place the calculated tendency of air
            potential temperature in the ``diagnostics`` output
            dictionary, ``False`` to regularly place it in the
            `tendencies` dictionary. Defaults to ``False``.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            ``True`` (respectively, ``False``) if the tendency
            of air potential temperature should be calculated at the
            interface (resp., main) vertical levels. Defaults to ``False``.
        air_pressure_on_interface_levels : `bool`, optional
            ``True`` (respectively, ``False``) if the input
            air potential pressure is defined at the interface
            (resp., main) vertical levels. Defaults to ``True``.
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
        physical_constants : `dict[str, sympl.DataArray]`, optional
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

        pcs = get_physical_constants(
            self._d_physical_constants, physical_constants
        )
        self._rd = pcs["gas_constant_of_dry_air"]
        self._cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {
            "height_on_interface_levels": {"dims": dims_stgz, "units": "m"}
        }

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
                dims = (
                    g.x.dims[0],
                    g.y.dims[0],
                    g.z_on_interface_levels.dims[0],
                )
                return_dict[
                    "air_potential_temperature_on_interface_levels"
                ] = {
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
                dims = (
                    g.x.dims[0],
                    g.y.dims[0],
                    g.z_on_interface_levels.dims[0],
                )
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
        dt = (
            (t - self._t0).total_seconds() / 3600.0
            if self._t0 is not None
            else t.hour
        )

        if dt <= 0.0:
            out = np.zeros(
                (mi, mj, mk), dtype=state["height_on_interface_levels"].dtype
            )
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
            p = (
                pv
                if pv.shape[2] == mk
                else 0.5 * (pv[:, :, 1:] + pv[:, :, :-1])
            )
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
                tendencies[
                    "air_potential_temperature_on_interface_levels"
                ] = out
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
