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
	ForwardEulerStepper
	BurgersStepper
	_ForwardEuler(BurgersStepper)
	_RK2(BurgersStepper)
	_RK3WS(BurgersStepper)
"""
import abc
import numpy as np

import gridtools as gt
from tasmania.python.burgers.dynamics.advection import BurgersAdvection

try:
	from tasmania.conf import nb as conf_nb
except ImportError:
	conf_nb = None


class ForwardEulerStep:
	def __init__(self, grid, advection):
		self._grid = grid
		self._advection = advection

	def __call__(self, dt, in_u, in_v, tmp_u, tmp_v, tnd_u=None, tnd_v=None):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# output fields
		out_u = gt.Equation()
		out_v = gt.Equation()

		# calculations
		adv_u_x, adv_u_y, adv_v_x, adv_v_y = self._advection(
			i, j, dx, dy, tmp_u, tmp_v
		)
		out_u[i, j] = in_u[i, j] - dt * (
			adv_u_x[i, j] + adv_u_y[i, j] if tnd_u is None
			else adv_u_x[i, j] + adv_u_y[i, j] - tnd_u[i, j]
		)
		out_v[i, j] = in_v[i, j] - dt * (
			adv_v_x[i, j] + adv_v_y[i, j] if tnd_v is None
			else adv_v_x[i, j] + adv_v_y[i, j] - tnd_v[i, j]
		)

		return out_u, out_v


class BurgersStepper:
	"""
	Abstract base class whose children integrate the 2-D inviscid Burgers
	equations implementing different time integrators.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid_xy, nb, flux_scheme, backend, dtype):
		"""
		Parameters
		----------
		grid_xy : tasmania.HorizontalGrid
			The underlying horizontal grid.
		nb : int
			Number of boundary layers.
		flux_scheme : str
			String specifying the advective flux scheme to be used.
			See :class:`tasmania.BurgersAdvection` for all available options.
		backend : obj
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated within
			this class.
		"""
		self._grid_xy = grid_xy
		self._backend = backend
		self._dtype = dtype

		self._advection = BurgersAdvection.factory(flux_scheme)

		assert nb >= self._advection.extent
		self._nb = nb

		self._fwe_step = ForwardEulerStep(grid_xy, self._advection)

	@property
	@abc.abstractmethod
	def stages(self):
		"""
		Returns
		-------
		int :
			Number of stages the time integrator consists of.
		"""
		pass

	@abc.abstractmethod
	def __call__(self, stage, state, tendency, timestep):
		"""
		Performing a stage of the time integrator.

		Parameters
		----------
		stage : int
			The stage to be performed.
		state : dict
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`numpy.ndarray`\s storing values
			for those variables.
		tendency : dict
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`numpy.ndarray`\s storing tendencies
			for those variables.
		timestep : timedelta
			The time step size.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting model variables,
			and whose values are :class:`numpy.ndarray`\s storing new values
			for those variables.
		"""
		pass

	@staticmethod
	def factory(
		time_integration_scheme, grid_xy, nb, flux_scheme, backend, dtype
	):
		"""
		Parameters
		----------
		time_integration_scheme : str
			String specifying the time integrator to be used. Either:

				* 'forward_euler' for the forward Euler method;
				* 'rk2' for the explicit, two-stages, second-order \
					Runge-Kutta (RK) method;
				* 'rk3ws' for the explicit, three-stages, second-order \
					RK method by Wicker & Skamarock.
		grid_xy : tasmania.HorizontalGrid
			The underlying horizontal grid.
		nb : int
			Number of boundary layers.
		flux_scheme : str
			String specifying the advective flux scheme to be used.
			See :class:`tasmania.BurgersAdvection` for all available options.
		backend : obj
			TODO
		dtype : numpy.dtype
			The data type for any :class:`numpy.ndarray` instantiated within
			this class.

		Return
		------
		tasmania.BurgersStepper :
			An instance of the appropriate derived class.
		"""
		args = (grid_xy, nb, flux_scheme, backend, dtype)
		if time_integration_scheme == 'forward_euler':
			return _ForwardEuler(*args)
		elif time_integration_scheme == 'rk2':
			return _RK2(*args)
		elif time_integration_scheme == 'rk3ws':
			return _RK3WS(*args)
		else:
			raise RuntimeError()


class _ForwardEuler(BurgersStepper):
	"""
	The forward Euler time integrator for the inviscid Burgers equations.
	"""
	def __init__(self, grid_xy, nb, flux_scheme, backend, dtype):
		super().__init__(grid_xy, nb, flux_scheme, backend, dtype)
		self._stencil = None

	@property
	def stages(self):
		return 1

	def __call__(self, stage, state, tendency, timestep):
		if self._stencil is None:
			self._stencil_initialize(tendency)

		self._dt.value = timestep.total_seconds()
		self._in_u[...] = state['x_velocity']
		self._in_v[...] = state['y_velocity']
		if 'x_velocity' in tendency:
			self._tnd_u[...] = tendency['x_velocity']
		if 'y_velocity' in tendency:
			self._tnd_v[...] = tendency['y_velocity']

		self._stencil.compute()

		return {
			'time': state['time'] + timestep,
			'x_velocity': self._out_u,
			'y_velocity': self._out_v,
		}

	def _stencil_initialize(self, tendency):
		mi, mj = self._grid_xy.nx, self._grid_xy.ny
		nb = self._nb

		self._dt = gt.Global()

		self._in_u = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._in_v = np.zeros((mi, mj, 1), dtype=self._dtype)
		inputs = {
			'in_u': self._in_u, 'in_v': self._in_v,
			'tmp_u': self._in_u, 'tmp_v': self._in_v
		}

		if 'x_velocity' in tendency:
			self._tnd_u = np.zeros((mi, mj, 1), dtype=self._dtype)
			inputs['tnd_u'] = self._tnd_u

		if 'y_velocity' in tendency:
			self._tnd_v = np.zeros((mi, mj, 1), dtype=self._dtype)
			inputs['tnd_v'] = self._tnd_v

		self._out_u = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._out_v = np.zeros((mi, mj, 1), dtype=self._dtype)
		outputs = {'out_u': self._out_u, 'out_v': self._out_v}

		self._stencil = gt.NGStencil(
			definitions_func=self._fwe_step.__call__,
			inputs=inputs, global_inputs={'dt': self._dt}, outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (mi-nb-1, mj-nb-1, 0)),
			mode=self._backend,
		)


class _RK2(BurgersStepper):
	"""
	The two-stages RK time integrator for the inviscid Burgers equations.
	"""
	def __init__(self, grid_xy, nb, flux_scheme, backend, dtype):
		super().__init__(grid_xy, nb, flux_scheme, backend, dtype)
		self._stencil = None

	@property
	def stages(self):
		return 2

	def __call__(self, stage, state, tendency, timestep):
		if self._stencil is None:
			self._stencil_initialize(tendency)

		if stage == 0:
			self._dt.value = 0.5 * timestep.total_seconds()
			self._in_u[...] = state['x_velocity'][...]
			self._in_v[...] = state['y_velocity'][...]
		elif stage == 1:
			self._dt.value = timestep.total_seconds()

		self._tmp_u[...] = state['x_velocity'][...]
		self._tmp_v[...] = state['y_velocity'][...]

		if 'x_velocity' in tendency:
			self._tnd_u[...] = tendency['x_velocity'][...]
		if 'y_velocity' in tendency:
			self._tnd_v[...] = tendency['y_velocity'][...]

		self._stencil.compute()

		return {
			'time': state['time'] + 0.5*timestep,
			'x_velocity': self._out_u,
			'y_velocity': self._out_v,
		}

	def _stencil_initialize(self, tendency):
		mi, mj = self._grid_xy.nx, self._grid_xy.ny
		nb = self._nb

		self._dt = gt.Global()

		self._in_u = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._in_v = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._tmp_u = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._tmp_v = np.zeros((mi, mj, 1), dtype=self._dtype)
		inputs = {
			'in_u': self._in_u, 'in_v': self._in_v,
			'tmp_u': self._tmp_u, 'tmp_v': self._tmp_v,
		}

		if 'x_velocity' in tendency:
			self._tnd_u = np.zeros((mi, mj, 1), dtype=self._dtype)
			inputs['tnd_u'] = self._tnd_u

		if 'y_velocity' in tendency:
			self._tnd_v = np.zeros((mi, mj, 1), dtype=self._dtype)
			inputs['tnd_v'] = self._tnd_v

		self._out_u = np.zeros((mi, mj, 1), dtype=self._dtype)
		self._out_v = np.zeros((mi, mj, 1), dtype=self._dtype)
		outputs = {'out_u': self._out_u, 'out_v': self._out_v}

		self._stencil = gt.NGStencil(
			definitions_func=self._fwe_step.__call__,
			inputs=inputs, global_inputs={'dt': self._dt}, outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (mi-nb-1, mj-nb-1, 0)),
			mode=self._backend,
		)


class _RK3WS(_RK2):
	"""
	The three-stages RK time integrator for the inviscid Burgers equations.
	"""
	def __init__(self, grid_xy, nb, flux_scheme, backend, dtype):
		super().__init__(grid_xy, nb, flux_scheme, backend, dtype)

	@property
	def stages(self):
		return 3

	def __call__(self, stage, state, tendency, timestep):
		if self._stencil is None:
			self._stencil_initialize(tendency)

		if stage == 0:
			dtr = 1.0/3.0 * timestep
			self._dt.value = 1.0/3.0 * timestep.total_seconds()
			self._in_u[...] = state['x_velocity'][...]
			self._in_v[...] = state['y_velocity'][...]
		elif stage == 1:
			dtr = 1.0/6.0 * timestep
			self._dt.value = 0.5 * timestep.total_seconds()
		else:
			dtr = 1.0/2.0 * timestep
			self._dt.value = timestep.total_seconds()

		self._tmp_u[...] = state['x_velocity'][...]
		self._tmp_v[...] = state['y_velocity'][...]

		if 'x_velocity' in tendency:
			self._tnd_u[...] = tendency['x_velocity'][...]
		if 'y_velocity' in tendency:
			self._tnd_v[...] = tendency['y_velocity'][...]

		self._stencil.compute()

		return {
			'time': state['time'] + dtr,
			'x_velocity': self._out_u,
			'y_velocity': self._out_v,
		}
