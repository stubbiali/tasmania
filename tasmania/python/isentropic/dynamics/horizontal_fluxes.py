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
"""
This module contains:
	IsentropicHorizontalFlux
	IsentropicNonconservativeHorizontalFlux
	IsentropicMinimalHorizontalFlux
	IsentropicBoussinesqMinimalHorizontalFlux
"""
import abc


class IsentropicHorizontalFlux(abc.ABC):
    """
	Abstract base class whose derived classes implement different schemes
	to compute the horizontal numerical fluxes for the three-dimensional
	isentropic dynamical core. The conservative form of the governing
	equations is used.
	"""

    # class attributes
    extent = None
    order = None
    externals = None

    @staticmethod
    @abc.abstractmethod
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        """
		This method returns the :class:`gridtools.Storage`\s representing
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
		s : gridtools.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		u : gridtools.Storage
			The x-staggered x-velocity, in units of [m s^-1].
		v : gridtools.Storage
			The y-staggered y-velocity, in units of [m s^-1].
		mtg : gridtools.Storage
			The Montgomery potential, in units of [m^2 s^-2].
		su : gridtools.Storage
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Storage
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.Storage`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.Storage`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.Storage`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		s_tnd : `gridtools.Storage`, optional
			The tendency of the isentropic density coming from physical parameterizations,
			in units of [kg m^-2 K^-1 s^-1].
		su_tnd : `gridtools.Storage`, optional
			The tendency of the x-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		sv_tnd : `gridtools.Storage`, optional
			The tendency of the y-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		qv_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of water vapor coming from physical
			parameterizations, in units of [g g^-1 s^-1].
		qc_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of cloud liquid water coming from
			physical parameterizations, in units of [g g^-1 s^-1].
		qr_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of precipitation water coming from
			physical parameterizations, in units of [g g^-1 s^-1].

		Returns
		-------
		flux_s_x : gridtools.Storage
			The x-flux for the isentropic density.
		flux_s_y : gridtools.Storage
			The y-flux for the isentropic density.
		flux_su_x : gridtools.Storage
			The x-flux for the x-momentum.
		flux_su_y : gridtools.Storage
			The y-flux for the x-momentum.
		flux_sv_x : gridtools.Storage
			The x-flux for the y-momentum.
		flux_sv_y : gridtools.Storage
			The y-flux for the y-momentum.
		flux_sqv_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of water vapor.
		flux_sqv_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of water vapor.
		flux_sqc_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of cloud liquid water.
		flux_sqc_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of cloud liquid water.
		flux_sqr_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of precipitation water.
		flux_sqr_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of precipitation water.
		"""

    @staticmethod
    def factory(scheme):
        """
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

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
			specified by :data:`scheme`.

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
            return MacCormack()
        elif scheme == "third_order_upwind":
            return ThirdOrderUpwind()
        elif scheme == "fifth_order_upwind":
            return FifthOrderUpwind()
        else:
            raise ValueError(
                "Unsupported horizontal flux scheme " "{}" "".format(scheme)
            )


class IsentropicNonconservativeHorizontalFlux(abc.ABC):
    """
	Abstract base class whose derived classes implement different schemes
	to compute the numerical fluxes for the three-dimensional isentropic
	dynamical core. The nonconservative form of the governing equations is used.
	"""

    # class attributes
    extent = None
    order = None
    externals = None

    @staticmethod
    @abc.abstractmethod
    def __call__(dt, dx, dy, s, u, v, mtg, qv=None, qc=None, qr=None):
        """
		Method returning the :class:`gridtools.Storage`\s representing the
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
		s : gridtools.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		u : gridtools.Storage
			The x-staggered x-velocity, in units of [m s^-1].
		v : gridtools.Storage
			The y-staggered y-velocity, in units of [m s^-1].
		mtg : gridtools.Storage
			The Montgomery potential, in units of [m^2 s^-2].
		qv : `gridtools.Storage`, optional
			The mass fraction of water vapor, in units of [g g^-1].
		qc : `gridtools.Storage`, optional
			The mass fraction of cloud liquid water, in units of [g g^-1].
		qr : `gridtools.Storage`, optional
			The mass fraction of precipitation water, in units of [g g^-1].

		Returns
		-------
		flux_s_x : gridtools.Storage
			The x-flux for the isentropic density.
		flux_s_y : gridtools.Storage
			The y-flux for the isentropic density.
		flux_u_x : gridtools.Storage
			The x-flux for the x-velocity.
		flux_u_y : gridtools.Storage
			The y-flux for the x-velocity.
		flux_v_x : gridtools.Storage
			The x-flux for the y-velocity.
		flux_v_y : gridtools.Storage
			The y-flux for the y-velocity.
		flux_qv_x : `gridtools.Storage`, optional
			The x-flux for the mass fraction of water vapor.
		flux_qv_y : `gridtools.Storage`, optional
			The y-flux for the mass fraction of water vapor.
		flux_qc_x : `gridtools.Storage`, optional
			The x-flux for the mass fraction of cloud liquid water.
		flux_qc_y : `gridtools.Storage`, optional
			The y-flux for the mass fraction of cloud liquid water.
		flux_qr_x : `gridtools.Storage`, optional
			The x-flux for the mass fraction of precipitation water.
		flux_qr_y : `gridtools.Storage`, optional
			The y-flux for the mass fraction of precipitation water.
		"""

    @staticmethod
    def factory(scheme):
        """
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

				* 'centered', for a second-order centered scheme.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
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
    extent = None
    order = None
    externals = None

    @staticmethod
    @abc.abstractmethod
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        """
		This method returns the :class:`gridtools.Storage`\s representing
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
		s : gridtools.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		u : gridtools.Storage
			The x-staggered x-velocity, in units of [m s^-1].
		v : gridtools.Storage
			The y-staggered y-velocity, in units of [m s^-1].
		su : gridtools.Storage
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Storage
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.Storage`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.Storage`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.Storage`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		s_tnd : `gridtools.Storage`, optional
			The tendency of the isentropic density coming from physical parameterizations,
			in units of [kg m^-2 K^-1 s^-1].
		su_tnd : `gridtools.Storage`, optional
			The tendency of the x-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		sv_tnd : `gridtools.Storage`, optional
			The tendency of the y-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		qv_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of water vapor coming from physical
			parameterizations, in units of [g g^-1 s^-1].
		qc_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of cloud liquid water coming from
			physical parameterizations, in units of [g g^-1 s^-1].
		qr_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of precipitation water coming from
			physical parameterizations, in units of [g g^-1 s^-1].

		Returns
		-------
		flux_s_x : gridtools.Storage
			The x-flux for the isentropic density.
		flux_s_y : gridtools.Storage
			The y-flux for the isentropic density.
		flux_su_x : gridtools.Storage
			The x-flux for the x-momentum.
		flux_su_y : gridtools.Storage
			The y-flux for the x-momentum.
		flux_sv_x : gridtools.Storage
			The x-flux for the y-momentum.
		flux_sv_y : gridtools.Storage
			The y-flux for the y-momentum.
		flux_sqv_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of water vapor.
		flux_sqv_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of water vapor.
		flux_sqc_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of cloud liquid water.
		flux_sqc_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of cloud liquid water.
		flux_sqr_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of precipitation water.
		flux_sqr_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of precipitation water.
		"""

    @staticmethod
    def factory(scheme):
        """
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

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
			specified by :data:`scheme`.

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
            raise ValueError(
                "Unsupported horizontal flux scheme " "{}" "".format(scheme)
            )


class IsentropicBoussinesqMinimalHorizontalFlux(abc.ABC):
    """
	Abstract base class whose derived classes implement different schemes
	to compute the horizontal numerical fluxes for the three-dimensional
	isentropic, Boussinesq and *minimal* dynamical core. The conservative
	form of the governing equations is used.
	"""

    # class attributes
    extent = None
    order = None
    externals = None

    @staticmethod
    @abc.abstractmethod
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        ddmtg,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        """
		This method returns the :class:`gridtools.Storage`\s representing
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
		s : gridtools.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		u : gridtools.Storage
			The x-staggered x-velocity, in units of [m s^-1].
		v : gridtools.Storage
			The y-staggered y-velocity, in units of [m s^-1].
		su : gridtools.Storage
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Storage
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		ddmtg : gridtools.Storage
			Second derivative with respect to the potential temperature
			of the Montgomery potential, in units of [m^2 K^-2 s^-2].
		sqv : `gridtools.Storage`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.Storage`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.Storage`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		s_tnd : `gridtools.Storage`, optional
			The tendency of the isentropic density coming from physical parameterizations,
			in units of [kg m^-2 K^-1 s^-1].
		su_tnd : `gridtools.Storage`, optional
			The tendency of the x-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		sv_tnd : `gridtools.Storage`, optional
			The tendency of the y-momentum coming from physical parameterizations,
			in units of [kg m^-1 K^-1 s^-2].
		qv_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of water vapor coming from physical
			parameterizations, in units of [g g^-1 s^-1].
		qc_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of cloud liquid water coming from
			physical parameterizations, in units of [g g^-1 s^-1].
		qr_tnd : `gridtools.Storage`, optional
			The tendency of the mass fraction of precipitation water coming from
			physical parameterizations, in units of [g g^-1 s^-1].

		Returns
		-------
		flux_s_x : gridtools.Storage
			The x-flux for the isentropic density.
		flux_s_y : gridtools.Storage
			The y-flux for the isentropic density.
		flux_su_x : gridtools.Storage
			The x-flux for the x-momentum.
		flux_su_y : gridtools.Storage
			The y-flux for the x-momentum.
		flux_sv_x : gridtools.Storage
			The x-flux for the y-momentum.
		flux_sv_y : gridtools.Storage
			The y-flux for the y-momentum.
		flux_ddmtg_x : gridtools.Storage
			The x-flux for the second derivative with respect to the 
			potential temperature of the Montgomery potential.
		flux_ddmtg_x : gridtools.Storage
			The y-flux for the second derivative with respect to the 
			potential temperature of the Montgomery potential.
		flux_sqv_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of water vapor.
		flux_sqv_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of water vapor.
		flux_sqc_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of cloud liquid water.
		flux_sqc_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of cloud liquid water.
		flux_sqr_x : `gridtools.Storage`, optional
			The x-flux for the isentropic density of precipitation water.
		flux_sqr_y : `gridtools.Storage`, optional
			The y-flux for the isentropic density of precipitation water.
		"""

    @staticmethod
    def factory(scheme):
        """
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

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
			specified by :data:`scheme`.

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
            raise ValueError(
                "Unsupported horizontal flux scheme " "{}" "".format(scheme)
            )
