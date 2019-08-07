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
	NGIsentropicMinimalVerticalFlux
	IsentropicBoussinesqMinimalVerticalFlux
"""
import abc


class IsentropicVerticalFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The conservative form of the governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	# class attributes
	extent = None
	order = None

	def __init__(self, grid, moist):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist = moist

	@abc.abstractmethod
	def __call__(
		self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
		sqv=None, sqv_prv=None, sqc=None,
		sqc_prv=None, sqr=None, sqr_prv=None
	):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the vertical fluxes for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : gridtools.Index
			The index running along the first horizontal dimension.
		j : gridtools.Index
			The index running along the second horizontal dimension.
		k : gridtools.Index
			The index running along the vertical dimension.
		dt : gridtools.Global
			The time step, in seconds.
		w : gridtools.Equation
			The vertical velocity, i.e., the change over time in potential temperature,
			in units of [K s^-1].
		s : gridtools.Equation
			The current isentropic density, in units of [kg m^-2 K^-1].
		s_prv : gridtools.Equation
			The provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-2 K^-1].
		su : gridtools.Equation
			The current x-momentum, in units of [kg m^-1 K^-1 s^-1].
		su_prv : gridtools.Equation
			The provisional x-momentum, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Equation
			The current y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv_prv : gridtools.Equation
			The provisional y-momentum, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.Equation`, optional
			The current isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqv_prv : `gridtools.Equation`, optional
			The provisional isentropic density of water vapor, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].
		sqc : `gridtools.Equation`, optional
			The current isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqc_prv : `gridtools.Equation`, optional
			The provisional isentropic density of cloud liquid water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].
		sqr : `gridtools.Equation`, optional
			The current isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		sqr_prv : `gridtools.Equation`, optional
			The provisional isentropic density of precipitation water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [kg m^-2 K^-1].

		Returns
		-------
		flux_s_z : gridtools.Equation
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.Equation
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.Equation
			The vertical flux for the y-momentum.
		flux_sqv_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist):
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

		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		from .implementations.vertical_fluxes import \
			Upwind, Centered, MacCormack
		if scheme == 'upwind':
			return Upwind(grid, moist)
		elif scheme == 'centered':
			return Centered(grid, moist)
		else:
			return MacCormack(grid, moist)


class IsentropicNonconservativeVerticalFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The nonconservative form of the governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	# class attributes
	extent = None
	order = None

	def __init__(self, grid, moist):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist = moist

	@abc.abstractmethod
	def __call__(
		self, i, j, k, dt, w, s, s_prv, u, u_prv, v, v_prv,
		qv=None, qv_prv=None, qc=None, qc_prv=None, qr=None, qr_prv=None
	):
		"""
		Method returning the :class:`gridtools.Equation`\s representing
		the vertical flux for all the prognostic model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : gridtools.Index
			The index running along the first horizontal dimension.
		j : gridtools.Index
			The index running along the second horizontal dimension.
		k : gridtools.Index
			The index running along the vertical dimension.
		dt : gridtools.Global
			The time step, in seconds.
		w : gridtools.Equation
			The vertical velocity, i.e., the change over time in potential temperature,
			in units of [K s^-1].
		s : gridtools.Equation
			The current isentropic density, in units of [kg m^-2 K^-1].
		s_prv : gridtools.Equation
			The provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [kg m^-2 K^-1].
		u : gridtools.Equation
			The current x-velocity, in units of [m s^-1].
		u_prv : gridtools.Equation
			The provisional x-velocity, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [m s^-1].
		v : gridtools.Equation
			The current y-velocity, in units of [m s^-1].
		v_prv : gridtools.Equation
			The provisional y-velocity, i.e., the isentropic density stepped
			disregarding the vertical advection, in units of [m s^-1].
		qv : `gridtools.Equation`, optional
			The current mass fraction of water vapor, in units of [g g^-1].
		qv_prv : `gridtools.Equation`, optional
			The provisional mass fraction of water vapor, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].
		qc : `gridtools.Equation`, optional
			The current mass fraction of cloud liquid water, in units of [g g^-1].
		qc_prv : `gridtools.Equation`, optional
			The provisional mass fraction of cloud liquid water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].
		qr : `gridtools.Equation`, optional
			The current mass fraction of precipitation water, in units of [g g^-1].
		qr_prv : `gridtools.Equation`, optional
			The provisional mass fraction of precipitation water, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection,
			in units of [g g^-1].

		Returns
		-------
		flux_s_z : gridtools.Equation
			The vertical flux for the isentropic density.
		flux_u_z : gridtools.Equation
			The vertical flux for the x-velocity.
		flux_v_z : gridtools.Equation
			The vertical flux for the y-velocity.
		flux_qv_z : `gridtools.Equation`, optional
			The vertical flux for the mass fraction of water vapor.
		flux_qc_z : `gridtools.Equation`, optional
			The vertical flux for the mass fraction of cloud liquid water.
		flux_qr_z : `gridtools.Equation`, optional
			The vertical flux for the mass fraction of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist):
		"""
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

				* 'centered', for a second-order centered scheme.

		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		from .implementations.nonconservative_vertical_fluxes import \
			Centered
		if scheme == 'centered':
			return Centered(grid, moist)


class IsentropicMinimalVerticalFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional
	isentropic and *minimal* dynamical core. The conservative form of the
	governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	# class attributes
	extent = None
	order = None

	def __init__(self, grid, moist):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist = moist

	@abc.abstractmethod
	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the vertical flux for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.
		Parameters
		----------
		k : gridtools.Index
			The index running along the vertical dimension.
		w : gridtools.Equation
			The vertical velocity, i.e., the change over time in potential temperature,
			defined at the vertical interface levels, in units of [K s^-1].
		s : gridtools.Equation
			The isentropic density, in units of [kg m^-2 K^-1].
		su : gridtools.Equation
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Equation
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		sqv : `gridtools.Equation`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.Equation`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.Equation`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].
		Returns
		-------
		flux_s_z : gridtools.Equation
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.Equation
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.Equation
			The vertical flux for the y-momentum.
		flux_sqv_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist):
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
		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
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
			return Upwind(grid, moist)
		elif scheme == 'centered':
			return Centered(grid, moist)
		elif scheme == 'third_order_upwind':
			return ThirdOrderUpwind(grid, moist)
		elif scheme == 'fifth_order_upwind':
			return FifthOrderUpwind(grid, moist)
		else:
			raise ValueError('Unsupported vertical flux scheme ''{}'''.format(scheme))


class NGIsentropicMinimalVerticalFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional
	isentropic and *minimal* dynamical core. The conservative form of the
	governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	# class attributes
	extent = None
	order = None

	def __init__(self, grid, tracers):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		tracers : ordered dict
			TODO
		"""
		self._grid = grid

		tracers = {} if tracers is None else tracers
		self._tracers = []
		for name, props in tracers.items():
			if 'stencil_symbol' not in props:
				import warnings
				warning_msg = \
					'Although not mandatory, it is advisable to specify ' \
					'a concise stencil symbol for {} to improve the readability ' \
					'of the generated code.'.format(name)
				warnings.warn(warning_msg, Warning)
			self._tracers.append(props.get('stencil_symbol', name))
			props['stencil_symbol'] = self._tracers[-1]

	@abc.abstractmethod
	def __call__(self, k, w, s, su, sv, **tracer_kwargs):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the vertical flux for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		k : gridtools.Index
			The index running along the vertical dimension.
		w : gridtools.Equation
			The vertical velocity, i.e., the change over time in potential temperature,
			defined at the vertical interface levels, in units of [K s^-1].
		s : gridtools.Equation
			The isentropic density, in units of [kg m^-2 K^-1].
		su : gridtools.Equation
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Equation
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		**tracer_kwargs:
			TODO

		Returns
		-------
		flux_s_z : gridtools.Equation
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.Equation
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.Equation
			The vertical flux for the y-momentum.
		"""

	@staticmethod
	def factory(scheme, grid, tracers=None):
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

		grid : tasmania.Grid
			The underlying grid.
		tracers : `ordered dict`, optional
			TODO

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
		from .implementations.ng_minimal_vertical_fluxes import \
			Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind
		if scheme == 'upwind':
			return Upwind(grid, tracers)
		elif scheme == 'centered':
			return Centered(grid, tracers)
		elif scheme == 'third_order_upwind':
			return ThirdOrderUpwind(grid, tracers)
		elif scheme == 'fifth_order_upwind':
			return FifthOrderUpwind(grid, tracers)
		else:
			raise ValueError('Unsupported vertical flux scheme ''{}'''.format(scheme))


class IsentropicBoussinesqMinimalVerticalFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional
	isentropic, Boussinesq and *minimal* dynamical core. The conservative
	form of the governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	# class attributes
	extent = None
	order = None

	def __init__(self, grid, moist):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist = moist

	@abc.abstractmethod
	def __call__(self, k, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the vertical flux for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		k : gridtools.Index
			The index running along the vertical dimension.
		w : gridtools.Equation
			The vertical velocity, i.e., the change over time in potential temperature,
			defined at the vertical interface levels, in units of [K s^-1].
		s : gridtools.Equation
			The isentropic density, in units of [kg m^-2 K^-1].
		su : gridtools.Equation
			The x-momentum, in units of [kg m^-1 K^-1 s^-1].
		sv : gridtools.Equation
			The y-momentum, in units of [kg m^-1 K^-1 s^-1].
		ddmtg : gridtools.Equation
			Second derivative with respect to the potential temperature
			of the Montgomery potential, in units of [m^2 K^-2 s^-2].
		sqv : `gridtools.Equation`, optional
			The isentropic density of water vapor, in units of [kg m^-2 K^-1].
		sqc : `gridtools.Equation`, optional
			The isentropic density of cloud liquid water, in units of [kg m^-2 K^-1].
		sqr : `gridtools.Equation`, optional
			The isentropic density of precipitation water, in units of [kg m^-2 K^-1].

		Returns
		-------
		flux_s_z : gridtools.Equation
			The vertical flux for the isentropic density.
		flux_su_z : gridtools.Equation
			The vertical flux for the x-momentum.
		flux_sv_z : gridtools.Equation
			The vertical flux for the y-momentum.
		flux_ddmtg_z : gridtools.Equation
			The vertical flux for the second derivative with respect to
			the potential temperature of the Montgomery potential.
		flux_sqv_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of water vapor.
		flux_sqc_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of cloud liquid water.
		flux_sqr_z : `gridtools.Equation`, optional
			The vertical flux for the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist):
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

		grid : tasmania.Grid
			The underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

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
			return Upwind(grid, moist)
		elif scheme == 'centered':
			return Centered(grid, moist)
		elif scheme == 'third_order_upwind':
			return ThirdOrderUpwind(grid, moist)
		elif scheme == 'fifth_order_upwind':
			return FifthOrderUpwind(grid, moist)
		else:
			raise ValueError('Unsupported vertical flux scheme ''{}'''.format(scheme))
