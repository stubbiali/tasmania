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
    HorizontalIsentropicFlux
    VerticalIsentropicFlux
    HorizontalNonconservativeIsentropicFlux
    VerticalNonconservativeIsentropicFlux
    HorizontalHomogeneousIsentropicFlux
"""
import abc


class HorizontalIsentropicFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the horizontal numerical fluxes for the three-dimensional isentropic
	dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist_on = moist_on

	@abc.abstractmethod
	def __call__(self, i, j, k, dt, s, u, v, mtg, su, sv, sqv=None, sqc=None, sqr=None,
				 u_tnd=None, v_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the :math:`x`- and :math:`y`-fluxes for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of water vapor.
		sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of cloud water.
		sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of precipitation water.
		u_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the :math:`x`-velocity.
		v_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the :math:`y`-velocity.
		qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of water vapor.
		qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of cloud liquid water.
		qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density.
		flux_su_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the :math:`x`-momentum.
		flux_su_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the :math:`x`-momentum.
		flux_sv_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the :math:`y`-momentum.
		flux_sv_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the :math:`y`-momentum.
		flux_sqv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of water vapor.
		flux_sqv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of water vapor.
		flux_sqc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of cloud liquid water.
		flux_sqc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of cloud liquid water.
		flux_sqr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of precipitation water.
		flux_sqr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist_on):
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

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
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
		import tasmania.dynamics._horizontal_isentropic_fluxes as module
		if scheme == 'upwind':
			return module._Upwind(grid, moist_on)
		elif scheme == 'centered':
			return module._Centered(grid, moist_on)
		elif scheme == 'maccormack':
			return module._MacCormack(grid, moist_on)
		elif scheme == 'third_order_upwind':
			return module._ThirdOrderUpwind(grid, moist_on)
		elif scheme == 'fifth_order_upwind':
			return module._FifthOrderUpwind(grid, moist_on)
		else:
			raise ValueError('Unsupported horizontal flux scheme ''{}'''.format(scheme))


class VerticalIsentropicFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist_on = moist_on

	@abc.abstractmethod
	def __call__(self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
				 sqv=None, sqv_prv=None, sqc=None,
				 sqc_prv=None, sqr=None, sqr_prv=None):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the :math:`\\theta`-fluxes for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity,
			i.e., the change over time in potential temperature.
		s : obj
			:class:`gridtools.Equation` representing the current
			isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional
			isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		su : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		su_prv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		sv : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		sv_prv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		sqv : `obj`, optional
			:class:`gridtools.Equation` representing the current
			isentropic density of water vapor.
		sqv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			isentropic density of water vapor, i.e., the isentropic
			density of water vapor stepped disregarding the vertical advection.
		sqc : `obj`, optional
			:class:`gridtools.Equation` representing the current
			isentropic density of cloud water.
		sqc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			isentropic density of cloud water, i.e., the isentropic
			density of cloud water stepped disregarding the vertical advection.
		sqr : `obj`, optional
			:class:`gridtools.Equation` representing the current
			isentropic density of precipitation water.
		sqr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			isentropic density of precipitation water, i.e., the isentropic
			density of precipitation water stepped disregarding
			the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the isentropic density.
		flux_su_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the :math:`x`-momentum.
		flux_sv_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the :math:`y`-momentum.
		flux_sqv_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the isentropic density of water vapor.
		flux_sqc_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the isentropic density of cloud water.
		flux_sqr_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist_on):
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

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		import tasmania.dynamics._vertical_isentropic_fluxes as module
		if scheme == 'upwind':
			return module._Upwind(grid, moist_on)
		elif scheme == 'centered':
			return module._Centered(grid, moist_on)
		else:
			return module._MacCormack(grid, moist_on)


class HorizontalNonconservativeIsentropicFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the numerical fluxes for the three-dimensional isentropic
	dynamical core. The nonconservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist_on = moist_on

	@abc.abstractmethod
	def __call__(self, i, j, k, dt, s, u, v, mtg, qv=None, qc=None, qr=None):
		"""
		Method returning the :class:`gridtools.Equation`\s representing the
		:math:`x`- and :math:`y`-fluxes for all the prognostic model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction
			of water vapour.
		qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction
			of cloud liquid water.
		qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass fraction
			of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the isentropic density.
		flux_u_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the :math:`x`-velocity.
		flux_u_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the :math:`x`-velocity.
		flux_v_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the :math:`y`-velocity.
		flux_v_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the :math:`y`-velocity.
		flux_qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the mass fraction of water vapor.
		flux_qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the mass fraction of water vapor.
		flux_qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the mass fraction of cloud liquid water.
		flux_qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the mass fraction of cloud liquid water.
		flux_qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux
			for the mass fraction of precipitation water.
		flux_qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux
			for the mass fraction of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist_on):
		"""
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

			* 'centered', for a second-order centered scheme.

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		import tasmania.dynamics._nonconservative_isentropic_fluxes as module
		if scheme == 'centered':
			return module._CenteredHorizontal(grid, moist_on)


class VerticalNonconservativeIsentropicFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the vertical numerical fluxes for the three-dimensional isentropic
	dynamical core. The nonconservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist_on = moist_on

	@abc.abstractmethod
	def __call__(self, i, j, k, dt, w, s, s_prv, u, u_prv, v, v_prv,
				 qv=None, qv_prv=None, qc=None, qc_prv=None, qr=None, qr_prv=None):
		"""
		Method returning the :class:`gridtools.Equation`\s representing
		the :math:`\\theta`-flux for all the prognostic model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		w : obj
			:class:`gridtools.Equation` representing the vertical
			velocity, i.e., the change over time in potential temperature.
		s : obj
			:class:`gridtools.Equation` representing the current
			isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional
			isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		u : obj
			:class:`gridtools.Equation` representing the current
			:math:`x`-velocity.
		u_prv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`x`-velocity, i.e., the :math:`x`-velocity stepped
			disregarding the vertical advection.
		v : obj
			:class:`gridtools.Equation` representing the current
			:math:`y`-velocity.
		v_prv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`y`-velocity, i.e., the :math:`y`-velocity stepped
			disregarding the vertical advection.
		qv : `obj`, optional
			:class:`gridtools.Equation` representing the current mass
			fraction of water vapor.
		qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			mass fraction of water vapor, i.e., the mass fraction of
			water vapor stepped disregarding the vertical advection.
		qc : `obj`, optional
			:class:`gridtools.Equation` representing the current mass
			fraction of cloud liquid water.
		qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			mass fraction of cloud liquid water, i.e., the mass fraction
			of cloud liquid water stepped disregarding the vertical advection.
		qr : `obj`, optional
			:class:`gridtools.Equation` representing the current mass
			fraction of precipitation water.
		qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional
			mass fraction of precipitation water, i.e., the mass fraction
			of precipitation water stepped disregarding the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the isentropic density.
		flux_u_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the :math:`x`-velocity.
		flux_v_z : obj
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the :math:`y`-velocity.
		flux_qv_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the mass fraction of water vapor.
		flux_qc_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the mass fraction of cloud liquid water.
		flux_qr_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`\\theta`-flux
			for the mass fraction of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist_on):
		"""
		Static method which returns an instance of the derived class
		implementing the numerical scheme specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

			* 'centered', for a second-order centered scheme.

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme
			specified by :data:`scheme`.
		"""
		import tasmania.dynamics._nonconservative_isentropic_fluxes as module
		if scheme == 'centered':
			return module._CenteredVertical(grid, moist_on)


class HorizontalHomogeneousIsentropicFlux:
	"""
	Abstract base class whose derived classes implement different schemes
	to compute the horizontal numerical fluxes for the three-dimensional
	isentropic and *homogeneous* dynamical core. The conservative form of the
	governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		self._grid = grid
		self._moist_on = moist_on

	@abc.abstractmethod
	def __call__(self, i, j, k, dt, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
				 u_tnd=None, v_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None):
		"""
		This method returns the :class:`gridtools.Equation`\s representing
		the :math:`x`- and :math:`y`-fluxes for all the conservative model variables.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of water vapor.
		sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of cloud water.
		sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of precipitation water.
		u_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the :math:`x`-velocity.
		v_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the :math:`y`-velocity.
		qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of water vapor.
		qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of cloud liquid water.
		qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass
			fraction of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density.
		flux_su_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the :math:`x`-momentum.
		flux_su_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the :math:`x`-momentum.
		flux_sv_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the :math:`y`-momentum.
		flux_sv_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the :math:`y`-momentum.
		flux_sqv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of water vapor.
		flux_sqv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of water vapor.
		flux_sqc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of cloud liquid water.
		flux_sqc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of cloud liquid water.
		flux_sqr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for
			the isentropic density of precipitation water.
		flux_sqr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for
			the isentropic density of precipitation water.
		"""

	@staticmethod
	def factory(scheme, grid, moist_on):
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

		grid : obj
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing
			the underlying grid.
		moist_on : bool
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
		import tasmania.dynamics._horizontal_homogeneous_isentropic_fluxes as module
		if scheme == 'upwind':
			return module._Upwind(grid, moist_on)
		elif scheme == 'centered':
			return module._Centered(grid, moist_on)
		elif scheme == 'maccormack':
			return module._MacCormack(grid, moist_on)
		elif scheme == 'third_order_upwind':
			return module._ThirdOrderUpwind(grid, moist_on)
		elif scheme == 'fifth_order_upwind':
			return module._FifthOrderUpwind(grid, moist_on)
		else:
			raise ValueError('Unsupported horizontal flux scheme ''{}'''.format(scheme))
