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
from typing import Any, Dict, List, Optional, Tuple

from gt4py import gtscript

from tasmania.python.framework.register import factorize
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_subroutine
from tasmania.python.utils import typing
from tasmania.python.utils.backend import is_gt


class IsentropicVerticalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the vertical numerical fluxes for the three-dimensional isentropic
    dynamical core. The conservative form of the governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None
    registry: Dict[str, "IsentropicVerticalFlux"] = {}

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dz: float,
        w: typing.gtfield_t,
        s: typing.gtfield_t,
        s_prv: typing.gtfield_t,
        su: typing.gtfield_t,
        su_prv: typing.gtfield_t,
        sv: typing.gtfield_t,
        sv_prv: typing.gtfield_t,
        sqv: "Optional[typing.gtfield_t]" = None,
        sqv_prv: "Optional[typing.gtfield_t]" = None,
        sqc: "Optional[typing.gtfield_t]" = None,
        sqc_prv: "Optional[typing.gtfield_t]" = None,
        sqr: "Optional[typing.gtfield_t]" = None,
        sqr_prv: "Optional[typing.gtfield_t]" = None,
    ) -> "Tuple[typing.gtfield_t, ...]":
        """
        This method returns the :class:`gt4py.gtscript.Field`\s representing
        the vertical fluxes for all the conservative model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dz : float
            The grid spacing in the vertical direction, in units of [K].
        w : gt4py.gtscript.Field
            The vertical velocity, i.e., the change over time in potential temperature,
            in units of [K s^-1].
        s : gt4py.gtscript.Field
            The current isentropic density, in units of [kg m^-2 K^-1].
        s_prv : gt4py.gtscript.Field
            The provisional isentropic density, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [kg m^-2 K^-1].
        su : gt4py.gtscript.Field
            The current x-momentum, in units of [kg m^-1 K^-1 s^-1].
        su_prv : gt4py.gtscript.Field
            The provisional x-momentum, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
        sv : gt4py.gtscript.Field
            The current y-momentum, in units of [kg m^-1 K^-1 s^-1].
        sv_prv : gt4py.gtscript.Field
            The provisional y-momentum, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
        sqv : `gt4py.gtscript.Field`, optional
            The current isentropic density of water vapor, in units of [kg m^-2 K^-1].
        sqv_prv : `gt4py.gtscript.Field`, optional
            The provisional isentropic density of water vapor, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [kg m^-2 K^-1].
        sqc : `gt4py.gtscript.Field`, optional
            The current isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
        sqc_prv : `gt4py.gtscript.Field`, optional
            The provisional isentropic density of cloud liquid water, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [kg m^-2 K^-1].
        sqr : `gt4py.gtscript.Field`, optional
            The current isentropic density of precipitation water, in units of [kg m^-2 K^-1].
        sqr_prv : `gt4py.gtscript.Field`, optional
            The provisional isentropic density of precipitation water, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [kg m^-2 K^-1].

        Returns
        -------
        flux_s_z : gt4py.gtscript.Field
            The vertical flux for the isentropic density.
        flux_su_z : gt4py.gtscript.Field
            The vertical flux for the x-momentum.
        flux_sv_z : gt4py.gtscript.Field
            The vertical flux for the y-momentum.
        flux_sqv_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of water vapor.
        flux_sqc_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of cloud liquid water.
        flux_sqr_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicVerticalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement. Either:

                * 'upwind', for the upwind scheme;
                * 'centered', for a second-order centered scheme;
                * 'maccormack', for the MacCormack scheme.

        Return
        ------
        obj :
            Instance of the derived class implementing the scheme
            specified by `scheme`.
        """
        return factorize(scheme, IsentropicVerticalFlux, ())


class IsentropicNonconservativeVerticalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the vertical numerical fluxes for the three-dimensional isentropic
    dynamical core. The nonconservative form of the governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None
    registry: Dict[str, "IsentropicNonconservativeVerticalFlux"] = {}

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dz: float,
        w: typing.gtfield_t,
        s: typing.gtfield_t,
        s_prv: typing.gtfield_t,
        u: typing.gtfield_t,
        u_prv: typing.gtfield_t,
        v: typing.gtfield_t,
        v_prv: typing.gtfield_t,
        qv: "Optional[typing.gtfield_t]" = None,
        qv_prv: "Optional[typing.gtfield_t]" = None,
        qc: "Optional[typing.gtfield_t]" = None,
        qc_prv: "Optional[typing.gtfield_t]" = None,
        qr: "Optional[typing.gtfield_t]" = None,
        qr_prv: "Optional[typing.gtfield_t]" = None,
    ) -> "Tuple[typing.gtfield_t, ...]":
        """
        Method returning the :class:`gt4py.gtscript.Field`\s representing
        the vertical flux for all the prognostic model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dz : float
            The grid spacing in the vertical direction, in units of [K].
        w : gt4py.gtscript.Field
            The vertical velocity, i.e., the change over time in potential temperature,
            in units of [K s^-1].
        s : gt4py.gtscript.Field
            The current isentropic density, in units of [kg m^-2 K^-1].
        s_prv : gt4py.gtscript.Field
            The provisional isentropic density, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [kg m^-2 K^-1].
        u : gt4py.gtscript.Field
            The current x-velocity, in units of [m s^-1].
        u_prv : gt4py.gtscript.Field
            The provisional x-velocity, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [m s^-1].
        v : gt4py.gtscript.Field
            The current y-velocity, in units of [m s^-1].
        v_prv : gt4py.gtscript.Field
            The provisional y-velocity, i.e., the isentropic density stepped
            disregarding the vertical advection, in units of [m s^-1].
        qv : `gt4py.gtscript.Field`, optional
            The current mass fraction of water vapor, in units of [g g^-1].
        qv_prv : `gt4py.gtscript.Field`, optional
            The provisional mass fraction of water vapor, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [g g^-1].
        qc : `gt4py.gtscript.Field`, optional
            The current mass fraction of cloud liquid water, in units of [g g^-1].
        qc_prv : `gt4py.gtscript.Field`, optional
            The provisional mass fraction of cloud liquid water, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [g g^-1].
        qr : `gt4py.gtscript.Field`, optional
            The current mass fraction of precipitation water, in units of [g g^-1].
        qr_prv : `gt4py.gtscript.Field`, optional
            The provisional mass fraction of precipitation water, i.e., the isentropic
            density of water vapor stepped disregarding the vertical advection,
            in units of [g g^-1].

        Returns
        -------
        flux_s_z : gt4py.gtscript.Field
            The vertical flux for the isentropic density.
        flux_u_z : gt4py.gtscript.Field
            The vertical flux for the x-velocity.
        flux_v_z : gt4py.gtscript.Field
            The vertical flux for the y-velocity.
        flux_qv_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the mass fraction of water vapor.
        flux_qc_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the mass fraction of cloud liquid water.
        flux_qr_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the mass fraction of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicNonconservativeVerticalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement. Either:

                * 'centered', for a second-order centered scheme.

        Return
        ------
        obj :
            Instance of the derived class implementing the scheme
            specified by `scheme`.
        """
        return factorize(scheme, IsentropicNonconservativeVerticalFlux, ())


class IsentropicMinimalVerticalFlux(StencilFactory, abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the vertical numerical fluxes for the three-dimensional
    isentropic and *minimal* dynamical core. The conservative form of the
    governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None
    registry: Dict[str, "IsentropicMinimalVerticalFlux"] = {}

    def __init__(self, backend):
        super().__init__(backend)

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dz: float,
        w: np.ndarray,
        s: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
    ) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dz: float,
        w: np.ndarray,
        sqv: np.ndarray,
        sqc: np.ndarray,
        sqr: np.ndarray,
    ) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dz: float,
        w: typing.gtfield_t,
        s: typing.gtfield_t,
        su: typing.gtfield_t,
        sv: typing.gtfield_t,
    ) -> "Tuple[typing.gtfield_t, ...]":
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dz: float,
        w: typing.gtfield_t,
        sqv: typing.gtfield_t,
        sqc: typing.gtfield_t,
        sqr: typing.gtfield_t,
    ) -> "Tuple[typing.gtfield_t, ...]":
        pass

    @staticmethod
    def factory(
        scheme: str, *, backend: str = "numpy"
    ) -> "IsentropicMinimalVerticalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement.
        backend : `str`, optional
            The backend.

        Return
        ------
        obj :
            Instance of the derived class implementing the scheme
            specified by `scheme`.

        References
        ----------
        Wicker, L. J., and W. C. Skamarock. (2002). Time-splitting methods for \
            elastic models using forward time schemes. *Monthly Weather Review*, \
            *130*:2088-2097.
        Zeman, C. (2016). An isentropic mountain flow model with iterative \
            synchronous flux correction. *Master thesis, ETH Zurich*.
        """
        return factorize(scheme, IsentropicMinimalVerticalFlux, (backend,))


class IsentropicBoussinesqMinimalVerticalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the vertical numerical fluxes for the three-dimensional
    isentropic, Boussinesq and *minimal* dynamical core. The conservative
    form of the governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None
    registry: Dict[str, "IsentropicBoussinesqMinimalVerticalFlux"] = {}

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dz: float,
        w: typing.gtfield_t,
        s: typing.gtfield_t,
        su: typing.gtfield_t,
        sv: typing.gtfield_t,
        ddmtg: typing.gtfield_t,
        sqv: "Optional[typing.gtfield_t]" = None,
        sqc: "Optional[typing.gtfield_t]" = None,
        sqr: "Optional[typing.gtfield_t]" = None,
    ) -> "Tuple[typing.gtfield_t, ...]":
        """
        This method returns the :class:`gt4py.gtscript.Field`\s representing
        the vertical flux for all the conservative model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dz : float
            The grid spacing in the vertical direction, in units of [K].
        w : gt4py.gtscript.Field
            The vertical velocity, i.e., the change over time in potential temperature,
            defined at the vertical interface levels, in units of [K s^-1].
        s : gt4py.gtscript.Field
            The isentropic density, in units of [kg m^-2 K^-1].
        su : gt4py.gtscript.Field
            The x-momentum, in units of [kg m^-1 K^-1 s^-1].
        sv : gt4py.gtscript.Field
            The y-momentum, in units of [kg m^-1 K^-1 s^-1].
        ddmtg : gt4py.gtscript.Field
            Second derivative with respect to the potential temperature
            of the Montgomery potential, in units of [m^2 K^-2 s^-2].
        sqv : `gt4py.gtscript.Field`, optional
            The isentropic density of water vapor, in units of [kg m^-2 K^-1].
        sqc : `gt4py.gtscript.Field`, optional
            The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
        sqr : `gt4py.gtscript.Field`, optional
            The isentropic density of precipitation water, in units of [kg m^-2 K^-1].

        Returns
        -------
        flux_s_z : gt4py.gtscript.Field
            The vertical flux for the isentropic density.
        flux_su_z : gt4py.gtscript.Field
            The vertical flux for the x-momentum.
        flux_sv_z : gt4py.gtscript.Field
            The vertical flux for the y-momentum.
        flux_ddmtg_z : gt4py.gtscript.Field
            The vertical flux for the second derivative with respect to
            the potential temperature of the Montgomery potential.
        flux_sqv_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of water vapor.
        flux_sqc_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of cloud liquid water.
        flux_sqr_z : `gt4py.gtscript.Field`, optional
            The vertical flux for the isentropic density of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicBoussinesqMinimalVerticalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement. Either:

                * 'upwind', for the upwind scheme;
                * 'centered', for a second-order centered scheme;
                * 'third_order_upwind', for the third-order upwind scheme;
                * 'fifth_order_upwind', for the fifth-order upwind scheme.

        Return
        ------
        obj :
            Instance of the derived class implementing the scheme
            specified by `scheme`.

        References
        ----------
        Wicker, L. J., and W. C. Skamarock. (2002). Time-splitting methods for \
            elastic models using forward time schemes. *Monthly Weather Review*, \
            *130*:2088-2097.
        Zeman, C. (2016). An isentropic mountain flow model with iterative \
            synchronous flux correction. *Master thesis, ETH Zurich*.
        """
        return factorize(scheme, IsentropicBoussinesqMinimalVerticalFlux, ())
