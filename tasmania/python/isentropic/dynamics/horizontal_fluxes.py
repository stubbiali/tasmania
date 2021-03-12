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
from tasmania.python.utils import typing as ty


class IsentropicHorizontalFlux(StencilFactory, abc.ABC):
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
    registry: Dict[str, "IsentropicHorizontalFlux"] = {}

    def __init__(self, backend: str) -> None:
        super().__init__(backend)

    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        self,
        dt: float,
        dx: float,
        dy: float,
        s: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
        mtg: np.ndarray,
        s_tnd: Optional[np.ndarray] = None,
        su_tnd: Optional[np.ndarray] = None,
        sv_tnd: Optional[np.ndarray] = None,
        *,
        compute_density_fluxes: bool = True,
        compute_momentum_fluxes: bool = True,
        compute_water_species_fluxes: bool = True
    ) -> List[np.ndarray]:
        pass

    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        self,
        dt: float,
        dx: float,
        dy: float,
        s: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        sqv: np.ndarray,
        sqc: np.ndarray,
        sqr: np.ndarray,
        qv_tnd: Optional[np.ndarray] = None,
        qc_tnd: Optional[np.ndarray] = None,
        qr_tnd: Optional[np.ndarray] = None,
        *,
        compute_water_species_fluxes: bool = True
    ) -> List[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        mtg: gtscript.Field["dtype"],
        s_tnd: "Optional[gtscript.Field['dtype']]" = None,
        su_tnd: "Optional[gtscript.Field['dtype']]" = None,
        sv_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        qv_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qc_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qr_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    def factory(
        scheme: str, *, backend: str = "numpy"
    ) -> "IsentropicHorizontalFlux":
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
        return factorize(scheme, IsentropicHorizontalFlux, (backend,))


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
    registry: Dict[str, "IsentropicNonconservativeHorizontalFlux"] = {}

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        mtg: gtscript.Field["dtype"],
        qv: "Optional[gtscript.Field['dtype']]" = None,
        qc: "Optional[gtscript.Field['dtype']]" = None,
        qr: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
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
        return factorize(scheme, IsentropicNonconservativeHorizontalFlux, ())


class IsentropicMinimalHorizontalFlux(StencilFactory, abc.ABC):
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
    registry: Dict[str, "IsentropicMinimalHorizontalFlux"] = {}

    def __init__(self, backend: str) -> None:
        super().__init__(backend)

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
        mtg: Optional[np.ndarray] = None,
        s_tnd: Optional[np.ndarray] = None,
        su_tnd: Optional[np.ndarray] = None,
        sv_tnd: Optional[np.ndarray] = None,
        *,
        compute_density_fluxes: bool = True,
        compute_momentum_fluxes: bool = True,
    ) -> List[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        sqv: np.ndarray,
        sqc: np.ndarray,
        sqr: np.ndarray,
        qv_tnd: Optional[np.ndarray] = None,
        qc_tnd: Optional[np.ndarray] = None,
        qr_tnd: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        mtg: "Optional[gtscript.Field['dtype']]" = None,
        s_tnd: "Optional[gtscript.Field['dtype']]" = None,
        su_tnd: "Optional[gtscript.Field['dtype']]" = None,
        sv_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        qv_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qc_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qr_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    def factory(
        scheme: str, *, backend: str = "numpy"
    ) -> "IsentropicMinimalHorizontalFlux":
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
        return factorize(scheme, IsentropicMinimalHorizontalFlux, (backend,))


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
    registry: Dict[str, "IsentropicBoussinesqMinimalHorizontalFlux"] = {}

    @staticmethod
    @gtscript.function
    @abc.abstractmethod
    def __call__(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        ddmtg: gtscript.Field["dtype"],
        sqv: "Optional[gtscript.Field['dtype']]" = None,
        sqc: "Optional[gtscript.Field['dtype']]" = None,
        sqr: "Optional[gtscript.Field['dtype']]" = None,
        s_tnd: "Optional[gtscript.Field['dtype']]" = None,
        su_tnd: "Optional[gtscript.Field['dtype']]" = None,
        sv_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qv_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qc_tnd: "Optional[gtscript.Field['dtype']]" = None,
        qr_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
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
        return factorize(scheme, IsentropicBoussinesqMinimalHorizontalFlux, ())
