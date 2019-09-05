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
	IsentropicVerticalFlux
	IsentropicNonconservativeVerticalFlux
	IsentropicMinimalVerticalFlux
	IsentropicBoussinesqMinimalVerticalFlux
"""
import abc


class IsentropicVerticalFlux(abc.ABC):
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The conservative form of the governing equations is used.
	"""
	# class attributes
	extent = None
	order = None
	externals = None

	@staticmethod
	@abc.abstractmethod
	def __call__(
		dt, dz, w, s, s_prv, su, su_prv, sv, sv_prv,
		sqv=None, sqv_prv=None, sqc=None, sqc_prv=None, sqr=None, sqr_prv=None
	):
		"""
		This method returns the :class:`gridtools.storage.Storage`\s representing
		the vertical fluxes for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		dt : float
			The time step, in seconds.
		dz : float
			The grid spacing in the vertical direction, in units of [K].
		w : gridtools.storage.Storage
			The vertical velocity, i.e., the change over time in potential temperature,
			in units of [K s^-1].
		s : gridtools.storage.Storage
			The current isentropic density, in units of [kg m^-2 K^-1].
		s_prv : gridtools.storage.Storage
			The provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-2 K^-1].
		su : gridtools.storage.Storage
			The current x-momentum, in units of [kg m^-1 K^-1 s^-1].
		su_prv : gridtools.storage.Storage
			The provisional x-momentum, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.storage.Storage
			The current y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv_prv : gridtools.storage.Storage
			The provisional y-momentum, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.storage.Storage`, optional
			The current isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqv_prv : `gridtools.storage.Storage`, optional
			The provisional isentropic density of water vapor, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].
		sqc : `gridtools.storage.Storage`, optional
			The current isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqc_prv : `gridtools.storage.Storage`, optional
			The provisional isentropic density of cloud liquid water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].
		sqr : `gridtools.storage.Storage`, optional
			The current isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		sqr_prv : `gridtools.storage.Storage`, optional
			The provisional isentropic density of precipitation water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].

		Returns
		-------
		flux_s_z : gridtools.storage.Storage
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.storage.Storage
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.storage.Storage
			The vertical flux for the y-momentum.
		flux_sqv_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of precipitation water.
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
				* 'maccormack', for the MacCormack scheme.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		from .implementations.vertical_fluxes import \
			Upwind, Centered, MacCormack
		if scheme == 'upwind':
			return Upwind()
		elif scheme == 'centered':
			return Centered()
		else:
			return MacCormack()


class IsentropicNonconservativeVerticalFlux(abc.ABC):
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The nonconservative form of the governing equations is used.
	"""
	# class attributes
	extent = None
	order = None
	externals = None

	@staticmethod
	@abc.abstractmethod
	def __call__(
		dt, dz, w, s, s_prv, u, u_prv, v, v_prv,
		qv=None, qv_prv=None, qc=None, qc_prv=None, qr=None, qr_prv=None
	):
		"""
		Method returning the :class:`gridtools.storage.Storage`\s representing
		the vertical flux for all the prognostic model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		dt : float
			The time step, in seconds.
		dz : float
			The grid spacing in the vertical direction, in units of [K].
		w : gridtools.storage.Storage
			The vertical velocity, i.e., the change over time in potential temperature,
			in units of [K s^-1].
		s : gridtools.storage.Storage
			The current isentropic density, in units of [kg m^-2 K^-1].
		s_prv : gridtools.storage.Storage
			The provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-2 K^-1].
		u : gridtools.storage.Storage
			The current x-velocity, in units of [m s^-1].
		u_prv : gridtools.storage.Storage
			The provisional x-velocity, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [m s^-1].
		v : gridtools.storage.Storage
			The current y-velocity, in units of [m s^-1].
		v_prv : gridtools.storage.Storage
			The provisional y-velocity, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [m s^-1].
		qv : `gridtools.storage.Storage`, optional
			The current mass fraction of water vapor, in units of [g g^-1].
		qv_prv : `gridtools.storage.Storage`, optional
			The provisional mass fraction of water vapor, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].
		qc : `gridtools.storage.Storage`, optional
			The current mass fraction of cloud liquid water, in units of [g g^-1].
		qc_prv : `gridtools.storage.Storage`, optional
			The provisional mass fraction of cloud liquid water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].
		qr : `gridtools.storage.Storage`, optional
			The current mass fraction of precipitation water, in units of [g g^-1].
		qr_prv : `gridtools.storage.Storage`, optional
			The provisional mass fraction of precipitation water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].

		Returns
		-------
		flux_s_z : gridtools.storage.Storage
			The vertical flux for the isentropic density.
		flux_u_z : gridtools.storage.Storage
			The vertical flux for the x-velocity.
		flux_v_z : gridtools.storage.Storage
			The vertical flux for the y-velocity.
		flux_qv_z : `gridtools.storage.Storage`, optional
			The vertical flux for the mass fraction of water vapor.
		flux_qc_z : `gridtools.storage.Storage`, optional
			The vertical flux for the mass fraction of cloud liquid water.
		flux_qr_z : `gridtools.storage.Storage`, optional
			The vertical flux for the mass fraction of precipitation water.
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
		from .implementations.nonconservative_vertical_fluxes import \
			Centered
		if scheme == 'centered':
			return Centered()


class IsentropicMinimalVerticalFlux(abc.ABC):
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional
	isentropic and *minimal* dynamical core. The conservative form of the
	governing equations is used.
	"""
	# class attributes
	extent = None
	order = None
	externals = None

	@staticmethod
	@abc.abstractmethod
	def __call__(dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		"""
		This method returns the :class:`gridtools.storage.Storage`\s representing
		the vertical flux for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		dt : float
			The time step, in seconds.
		dz : float
			The grid spacing in the vertical direction, in units of [K].
		w : gridtools.storage.Storage
			The vertical velocity, i.e., the change over time in potential temperature,
			defined at the vertical interface levels, in units of [K s^-1].
		s : gridtools.storage.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		su : gridtools.storage.Storage
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.storage.Storage
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.storage.Storage`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.storage.Storage`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.storage.Storage`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].

		Returns
		-------
		flux_s_z : gridtools.storage.Storage
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.storage.Storage
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.storage.Storage
			The vertical flux for the y-momentum.
		flux_sqv_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of precipitation water.
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
		from .implementations.minimal_vertical_fluxes import \
			Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind
		if scheme == 'upwind':
			return Upwind()
		elif scheme == 'centered':
			return Centered()
		elif scheme == 'third_order_upwind':
			return ThirdOrderUpwind()
		elif scheme == 'fifth_order_upwind':
			return FifthOrderUpwind()
		else:
			raise ValueError('Unsupported vertical flux scheme ''{}'''.format(scheme))


class IsentropicBoussinesqMinimalVerticalFlux(abc.ABC):
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional
	isentropic, Boussinesq and *minimal* dynamical core. The conservative
	form of the governing equations is used.
	"""
	# class attributes
	extent = None
	order = None
	externals = None

	@staticmethod
	@abc.abstractmethod
	def __call__(dt, dz, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		"""
		This method returns the :class:`gridtools.storage.Storage`\s representing
		the vertical flux for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		dt : float
			The time step, in seconds.
		dz : float
			The grid spacing in the vertical direction, in units of [K].
		w : gridtools.storage.Storage
			The vertical velocity, i.e., the change over time in potential temperature,
			defined at the vertical interface levels, in units of [K s^-1].
		s : gridtools.storage.Storage
			The isentropic density, in units of [kg m^-2 K^-1].
		su : gridtools.storage.Storage
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.storage.Storage
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		ddmtg : gridtools.storage.Storage
			Second derivative with respect to the potential temperature
			of the Montgomery potential, in units of [m^2 K^-2 s^-2].
		sqv : `gridtools.storage.Storage`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.storage.Storage`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.storage.Storage`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].

		Returns
		-------
		flux_s_z : gridtools.storage.Storage
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.storage.Storage
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.storage.Storage
			The vertical flux for the y-momentum.
		flux_ddmtg_z : gridtools.storage.Storage
			The vertical flux for the second derivative with respect to
			the potential temperature of the Montgomery potential.
		flux_sqv_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.storage.Storage`, optional
			The vertical flux for the isentropic density of precipitation water.
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
		from .implementations.boussinesq_minimal_vertical_fluxes import \
			Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind
		if scheme == 'upwind':
			return Upwind()
		elif scheme == 'centered':
			return Centered()
		elif scheme == 'third_order_upwind':
			return ThirdOrderUpwind()
		elif scheme == 'fifth_order_upwind':
			return FifthOrderUpwind()
		else:
			raise ValueError('Unsupported vertical flux scheme ''{}'''.format(scheme))
