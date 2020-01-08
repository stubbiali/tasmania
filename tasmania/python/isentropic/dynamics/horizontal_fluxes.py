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
from typing import Any, Dict, Optional, Tuple

from gt4py import gtscript

from tasmania.python.utils import taz_types


class IsentropicHorizontalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the horizontal numerical fluxes for the three-dimensional
    isentropic dynamical core. The conservative form of the governing
    equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: taz_types.gtfield_t,
        u: taz_types.gtfield_t,
        v: taz_types.gtfield_t,
        su: taz_types.gtfield_t,
        sv: taz_types.gtfield_t,
        mtg: taz_types.gtfield_t,
        sqv: "Optional[taz_types.gtfield_t]" = None,
        sqc: "Optional[taz_types.gtfield_t]" = None,
        sqr: "Optional[taz_types.gtfield_t]" = None,
        s_tnd: "Optional[taz_types.gtfield_t]" = None,
        su_tnd: "Optional[taz_types.gtfield_t]" = None,
        sv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qc_tnd: "Optional[taz_types.gtfield_t]" = None,
        qr_tnd: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":
        """
        This method returns the :class:`gt4py.gtscript.Field`\s representing
        the x- and y-fluxes for all the conservative model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dx : float
            The grid spacing in the x-direction, in meters.
        dy : float
            The grid spacing in the y-direction, in meters.
        s : gt4py.gtscript.Field
            The isentropic density, in units of [kg m^-2 K^-1].
        u : gt4py.gtscript.Field
            The x-staggered x-velocity, in units of [m s^-1].
        v : gt4py.gtscript.Field
            The y-staggered y-velocity, in units of [m s^-1].
        mtg : gt4py.gtscript.Field
            The Montgomery potential, in units of [m^2 s^-2].
        su : gt4py.gtscript.Field
            The x-momentum, in units of [kg m^-1 K^-1 s^-1].
        sv : gt4py.gtscript.Field
            The y-momentum, in units of [kg m^-1 K^-1 s^-1].
        sqv : `gt4py.gtscript.Field`, optional
            The isentropic density of water vapor, in units of [kg m^-2 K^-1].
        sqc : `gt4py.gtscript.Field`, optional
            The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
        sqr : `gt4py.gtscript.Field`, optional
            The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
        s_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the isentropic density coming from physical parameterizations,
            in units of [kg m^-2 K^-1 s^-1].
        su_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the x-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        sv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the y-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        qv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of water vapor coming from physical
            parameterizations, in units of [g g^-1 s^-1].
        qc_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of cloud liquid water coming from
            physical parameterizations, in units of [g g^-1 s^-1].
        qr_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of precipitation water coming from
            physical parameterizations, in units of [g g^-1 s^-1].

        Returns
        -------
        flux_s_x : gt4py.gtscript.Field
            The x-flux for the isentropic density.
        flux_s_y : gt4py.gtscript.Field
            The y-flux for the isentropic density.
        flux_su_x : gt4py.gtscript.Field
            The x-flux for the x-momentum.
        flux_su_y : gt4py.gtscript.Field
            The y-flux for the x-momentum.
        flux_sv_x : gt4py.gtscript.Field
            The x-flux for the y-momentum.
        flux_sv_y : gt4py.gtscript.Field
            The y-flux for the y-momentum.
        flux_sqv_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of water vapor.
        flux_sqv_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of water vapor.
        flux_sqc_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of cloud liquid water.
        flux_sqc_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of cloud liquid water.
        flux_sqr_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of precipitation water.
        flux_sqr_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicHorizontalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement. Either:

                * 'upwind', for the upwind scheme;
                * 'centered', for a second-order centered scheme;
                * 'maccormack', for the MacCormack scheme;
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
        from .implementations.horizontal_fluxes import (
            Upwind,
            Centered,
            MacCormack,
            ThirdOrderUpwind,
            FifthOrderUpwind,
        )

        if scheme == "upwind":
            return Upwind()
        elif scheme == "centered":
            return Centered()
        elif scheme == "maccormack":
            raise NotImplementedError
            # return MacCormack()
        elif scheme == "third_order_upwind":
            return ThirdOrderUpwind()
        elif scheme == "fifth_order_upwind":
            return FifthOrderUpwind()
        else:
            raise ValueError("Unsupported horizontal flux scheme " "{}" "".format(scheme))


class IsentropicNonconservativeHorizontalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the numerical fluxes for the three-dimensional isentropic
    dynamical core. The nonconservative form of the governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: taz_types.gtfield_t,
        u: taz_types.gtfield_t,
        v: taz_types.gtfield_t,
        mtg: taz_types.gtfield_t,
        qv: "Optional[taz_types.gtfield_t]" = None,
        qc: "Optional[taz_types.gtfield_t]" = None,
        qr: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":
        """
        Method returning the :class:`gt4py.gtscript.Field`\s representing the
        x- and y-fluxes for all the prognostic model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dx : float
            The grid spacing in the x-direction, in meters.
        dy : float
            The grid spacing in the y-direction, in meters.
        s : gt4py.gtscript.Field
            The isentropic density, in units of [kg m^-2 K^-1].
        u : gt4py.gtscript.Field
            The x-staggered x-velocity, in units of [m s^-1].
        v : gt4py.gtscript.Field
            The y-staggered y-velocity, in units of [m s^-1].
        mtg : gt4py.gtscript.Field
            The Montgomery potential, in units of [m^2 s^-2].
        qv : `gt4py.gtscript.Field`, optional
            The mass fraction of water vapor, in units of [g g^-1].
        qc : `gt4py.gtscript.Field`, optional
            The mass fraction of cloud liquid water, in units of [g g^-1].
        qr : `gt4py.gtscript.Field`, optional
            The mass fraction of precipitation water, in units of [g g^-1].

        Returns
        -------
        flux_s_x : gt4py.gtscript.Field
            The x-flux for the isentropic density.
        flux_s_y : gt4py.gtscript.Field
            The y-flux for the isentropic density.
        flux_u_x : gt4py.gtscript.Field
            The x-flux for the x-velocity.
        flux_u_y : gt4py.gtscript.Field
            The y-flux for the x-velocity.
        flux_v_x : gt4py.gtscript.Field
            The x-flux for the y-velocity.
        flux_v_y : gt4py.gtscript.Field
            The y-flux for the y-velocity.
        flux_qv_x : `gt4py.gtscript.Field`, optional
            The x-flux for the mass fraction of water vapor.
        flux_qv_y : `gt4py.gtscript.Field`, optional
            The y-flux for the mass fraction of water vapor.
        flux_qc_x : `gt4py.gtscript.Field`, optional
            The x-flux for the mass fraction of cloud liquid water.
        flux_qc_y : `gt4py.gtscript.Field`, optional
            The y-flux for the mass fraction of cloud liquid water.
        flux_qr_x : `gt4py.gtscript.Field`, optional
            The x-flux for the mass fraction of precipitation water.
        flux_qr_y : `gt4py.gtscript.Field`, optional
            The y-flux for the mass fraction of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicNonconservativeHorizontalFlux":
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
        from .implementations.nonconservative_horizontal_fluxes import Centered

        if scheme == "centered":
            return Centered()


class IsentropicMinimalHorizontalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the horizontal numerical fluxes for the three-dimensional
    isentropic and *minimal* dynamical core. The conservative form of the
    governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: taz_types.gtfield_t,
        u: taz_types.gtfield_t,
        v: taz_types.gtfield_t,
        su: taz_types.gtfield_t,
        sv: taz_types.gtfield_t,
        mtg: "Optional[taz_types.gtfield_t]" = None,
        sqv: "Optional[taz_types.gtfield_t]" = None,
        sqc: "Optional[taz_types.gtfield_t]" = None,
        sqr: "Optional[taz_types.gtfield_t]" = None,
        s_tnd: "Optional[taz_types.gtfield_t]" = None,
        su_tnd: "Optional[taz_types.gtfield_t]" = None,
        sv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qc_tnd: "Optional[taz_types.gtfield_t]" = None,
        qr_tnd: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":
        """
        This method returns the :class:`gt4py.gtscript.Field`\s representing
        the x- and y-fluxes for all the conservative model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dx : float
            The grid spacing in the x-direction, in meters.
        dy : float
            The grid spacing in the y-direction, in meters.
        s : gt4py.gtscript.Field
            The isentropic density, in units of [kg m^-2 K^-1].
        u : gt4py.gtscript.Field
            The x-staggered x-velocity, in units of [m s^-1].
        v : gt4py.gtscript.Field
            The y-staggered y-velocity, in units of [m s^-1].
        su : gt4py.gtscript.Field
            The x-momentum, in units of [kg m^-1 K^-1 s^-1].
        sv : gt4py.gtscript.Field
            The y-momentum, in units of [kg m^-1 K^-1 s^-1].
        sqv : `gt4py.gtscript.Field`, optional
            The isentropic density of water vapor, in units of [kg m^-2 K^-1].
        sqc : `gt4py.gtscript.Field`, optional
            The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
        sqr : `gt4py.gtscript.Field`, optional
            The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
        s_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the isentropic density coming from physical parameterizations,
            in units of [kg m^-2 K^-1 s^-1].
        su_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the x-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        sv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the y-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        qv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of water vapor coming from physical
            parameterizations, in units of [g g^-1 s^-1].
        qc_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of cloud liquid water coming from
            physical parameterizations, in units of [g g^-1 s^-1].
        qr_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of precipitation water coming from
            physical parameterizations, in units of [g g^-1 s^-1].

        Returns
        -------
        flux_s_x : gt4py.gtscript.Field
            The x-flux for the isentropic density.
        flux_s_y : gt4py.gtscript.Field
            The y-flux for the isentropic density.
        flux_su_x : gt4py.gtscript.Field
            The x-flux for the x-momentum.
        flux_su_y : gt4py.gtscript.Field
            The y-flux for the x-momentum.
        flux_sv_x : gt4py.gtscript.Field
            The x-flux for the y-momentum.
        flux_sv_y : gt4py.gtscript.Field
            The y-flux for the y-momentum.
        flux_sqv_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of water vapor.
        flux_sqv_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of water vapor.
        flux_sqc_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of cloud liquid water.
        flux_sqc_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of cloud liquid water.
        flux_sqr_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of precipitation water.
        flux_sqr_y : `gt4py.gtscript.Field`, optional
            The y-flux for the isentropic density of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicMinimalHorizontalFlux":
        """
        Static method which returns an instance of the derived class
        implementing the numerical scheme specified by `scheme`.

        Parameters
        ----------
        scheme : str
            String specifying the numerical scheme to implement. Either:

                * 'upwind', for the upwind scheme;
                * 'centered', for a second-order centered scheme;
                * 'maccormack', for the MacCormack scheme;
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
        from .implementations.minimal_horizontal_fluxes import (
            Upwind,
            Centered,
            MacCormack,
            ThirdOrderUpwind,
            FifthOrderUpwind,
        )

        if scheme == "upwind":
            return Upwind()
        elif scheme == "centered":
            return Centered()
        elif scheme == "maccormack":
            return MacCormack()
        elif scheme == "third_order_upwind":
            return ThirdOrderUpwind()
        elif scheme == "fifth_order_upwind":
            return FifthOrderUpwind()
        else:
            raise ValueError("Unsupported horizontal flux scheme " "{}" "".format(scheme))


class IsentropicBoussinesqMinimalHorizontalFlux(abc.ABC):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the horizontal numerical fluxes for the three-dimensional
    isentropic, Boussinesq and *minimal* dynamical core. The conservative
    form of the governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: taz_types.gtfield_t,
        u: taz_types.gtfield_t,
        v: taz_types.gtfield_t,
        su: taz_types.gtfield_t,
        sv: taz_types.gtfield_t,
        ddmtg: taz_types.gtfield_t,
        sqv: "Optional[taz_types.gtfield_t]" = None,
        sqc: "Optional[taz_types.gtfield_t]" = None,
        sqr: "Optional[taz_types.gtfield_t]" = None,
        s_tnd: "Optional[taz_types.gtfield_t]" = None,
        su_tnd: "Optional[taz_types.gtfield_t]" = None,
        sv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qc_tnd: "Optional[taz_types.gtfield_t]" = None,
        qr_tnd: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":
        """
        This method returns the :class:`gt4py.gtscript.Field`\s representing
        the x- and y-fluxes for all the conservative model variables.
        As this method is marked as abstract, its implementation is delegated
        to the derived classes.

        Parameters
        ----------
        dt : float
            The time step, in seconds.
        dx : float
            The grid spacing in the x-direction, in meters.
        dy : float
            The grid spacing in the y-direction, in meters.
        s : gt4py.gtscript.Field
            The isentropic density, in units of [kg m^-2 K^-1].
        u : gt4py.gtscript.Field
            The x-staggered x-velocity, in units of [m s^-1].
        v : gt4py.gtscript.Field
            The y-staggered y-velocity, in units of [m s^-1].
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
        s_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the isentropic density coming from physical parameterizations,
            in units of [kg m^-2 K^-1 s^-1].
        su_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the x-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        sv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the y-momentum coming from physical parameterizations,
            in units of [kg m^-1 K^-1 s^-2].
        qv_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of water vapor coming from physical
            parameterizations, in units of [g g^-1 s^-1].
        qc_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of cloud liquid water coming from
            physical parameterizations, in units of [g g^-1 s^-1].
        qr_tnd : `gt4py.gtscript.Field`, optional
            The tendency of the mass fraction of precipitation water coming from
            physical parameterizations, in units of [g g^-1 s^-1].

        Returns
        -------
        flux_s_x : gt4py.gtscript.Field
            The x-flux for the isentropic density.
        flux_s_y : gt4py.gtscript.Field
            The y-flux for the isentropic density.
        flux_su_x : gt4py.gtscript.Field
            The x-flux for the x-momentum.
        flux_su_y : gt4py.gtscript.Field
            The y-flux for the x-momentum.
        flux_sv_x : gt4py.gtscript.Field
            The x-flux for the y-momentum.
        flux_sv_y : gt4py.gtscript.Field
            The y-flux for the y-momentum.
        flux_ddmtg_x : gt4py.gtscript.Field
            The x-flux for the second derivative with respect to the
            potential temperature of the Montgomery potential.
        flux_ddmtg_x : gt4py.gtscript.Field
            The y-flux for the second derivative with respect to the
            potential temperature of the Montgomery potential.
        flux_sqv_x : `gt4py.gtscript.Field`, optional
            The x-flux for the isentropic density of water vapor.
        flux_sqv_y : `gt4py.storage.Storage`, optional
            The y-flux for the isentropic density of water vapor.
        flux_sqc_x : `gt4py.storage.Storage`, optional
            The x-flux for the isentropic density of cloud liquid water.
        flux_sqc_y : `gt4py.storage.Storage`, optional
            The y-flux for the isentropic density of cloud liquid water.
        flux_sqr_x : `gt4py.storage.Storage`, optional
            The x-flux for the isentropic density of precipitation water.
        flux_sqr_y : `gt4py.storage.Storage`, optional
            The y-flux for the isentropic density of precipitation water.
        """
        pass

    @staticmethod
    def factory(scheme: str) -> "IsentropicBoussinesqMinimalHorizontalFlux":
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
        from .implementations.boussinesq_minimal_horizontal_fluxes import (
            Upwind,
            Centered,
            ThirdOrderUpwind,
            FifthOrderUpwind,
        )

        if scheme == "upwind":
            return Upwind()
        elif scheme == "centered":
            return Centered()
        elif scheme == "third_order_upwind":
            return ThirdOrderUpwind()
        elif scheme == "fifth_order_upwind":
            return FifthOrderUpwind()
        else:
            raise ValueError("Unsupported horizontal flux scheme " "{}" "".format(scheme))
