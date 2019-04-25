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
	BurgersAdvection
	_FirstOrder(BurgersAdvection)
	_SecondOrder(BurgersAdvection)
	_ThirdOrder(BurgersAdvection)
	_FourthOrder(BurgersAdvection)
	_FifthOrder(BurgersAdvection)
	_SixthOrder(BurgersAdvection)
"""
import abc

import gridtools as gt


class BurgersAdvection:
	"""
	A discretizer for the 2-D Burgers advection flux.
	"""
	__metaclass__ = abc.ABCMeta

	extent = None

	@staticmethod
	@abc.abstractmethod
	def __call__(i, j, dx, dy, u, v):
		pass

	@staticmethod
	def factory(flux_scheme):
		if flux_scheme == 'first_order':
			return _FirstOrder()
		elif flux_scheme == 'second_order':
			return _SecondOrder()
		elif flux_scheme == 'third_order':
			return _ThirdOrder()
		elif flux_scheme == 'fourth_order':
			return _FourthOrder()
		elif flux_scheme == 'fifth_order':
			return _FifthOrder()
		elif flux_scheme == 'sixth_order':
			return _SixthOrder()
		else:
			raise RuntimeError()


class _FirstOrder(BurgersAdvection):
	extent = 1

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		abs_u = gt.Equation()
		abs_v = gt.Equation()

		abs_u[i, j] = u[i, j] * (u[i, j] >= 0.0) - u[i, j] * (u[i, j] < 0)
		abs_v[i, j] = v[i, j] * (v[i, j] >= 0.0) - v[i, j] * (v[i, j] < 0)

		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (2.0 * dx) * (u[i+1, j] - u[i-1, j]) - \
			abs_u[i, j] / (2.0 * dx) * (u[i+1, j] - 2.0*u[i, j] + u[i-1, j])
		adv_u_y[i, j] = v[i, j] / (2.0 * dy) * (u[i, j+1] - u[i, j-1]) - \
			abs_v[i, j] / (2.0 * dy) * (u[i, j+1] - 2.0*u[i, j] + u[i, j-1])
		adv_v_x[i, j] = u[i, j] / (2.0 * dx) * (v[i+1, j] - v[i-1, j]) - \
			abs_u[i, j] / (2.0 * dx) * (v[i+1, j] - 2.0*v[i, j] + v[i-1, j])
		adv_v_y[i, j] = v[i, j] / (2.0 * dy) * (v[i, j+1] - v[i, j-1]) - \
			abs_v[i, j] / (2.0 * dy) * (v[i, j+1] - 2.0*v[i, j] + v[i, j-1])

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _SecondOrder(BurgersAdvection):
	extent = 1

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (2.0 * dx) * (u[i+1, j] - u[i-1, j])
		adv_u_y[i, j] = v[i, j] / (2.0 * dy) * (u[i, j+1] - u[i, j-1])
		adv_v_x[i, j] = u[i, j] / (2.0 * dx) * (v[i+1, j] - v[i-1, j])
		adv_v_y[i, j] = v[i, j] / (2.0 * dy) * (v[i, j+1] - v[i, j-1])

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _ThirdOrder(BurgersAdvection):
	extent = 2

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		abs_u = gt.Equation()
		abs_v = gt.Equation()

		abs_u[i, j] = u[i, j] * (u[i, j] >= 0.0) - u[i, j] * (u[i, j] < 0)
		abs_v[i, j] = v[i, j] * (v[i, j] >= 0.0) - v[i, j] * (v[i, j] < 0)

		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (12.0 * dx) * (
				8.0 * (u[i+1, j] - u[i-1, j]) - (u[i+2, j] - u[i-2, j])
			) \
			+ abs_u[i, j] / (12.0 * dx) * (
				u[i+2, j] - 4.0*u[i+1, j] + 6.0*u[i, j] - 4.0*u[i-1, j] + u[i-2, j]
			)
		adv_u_y[i, j] = v[i, j] / (12.0 * dy) * (
				8.0 * (u[i, j+1] - u[i, j-1]) - (u[i, j+2] - u[i, j-2])
			) \
			+ abs_v[i, j] / (12.0 * dy) * (
				u[i, j+2] - 4.0*u[i, j+1] + 6.0*u[i, j] - 4.0*u[i, j-1] + u[i, j-2]
			)
		adv_v_x[i, j] = u[i, j] / (12.0 * dx) * (
				8.0 * (v[i+1, j] - v[i-1, j]) - (v[i+2, j] - v[i-2, j])
			) \
			+ abs_u[i, j] / (12.0 * dx) * (
				v[i+2, j] - 4.0*v[i+1, j] + 6.0*v[i, j] - 4.0*v[i-1, j] + v[i-2, j]
			)
		adv_v_y[i, j] = v[i, j] / (12.0 * dy) * (
				8.0 * (v[i, j+1] - v[i, j-1]) - (v[i, j+2] - v[i, j-2])
			) \
			+ abs_v[i, j] / (12.0 * dy) * (
				v[i, j+2] - 4.0*v[i, j+1] + 6.0*v[i, j] - 4.0*v[i, j-1] + v[i, j-2]
			)

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _FourthOrder(BurgersAdvection):
	extent = 2

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (12.0 * dx) * (
				8.0 * (u[i+1, j] - u[i-1, j]) - (u[i+2, j] - u[i-2, j])
			)
		adv_u_y[i, j] = v[i, j] / (12.0 * dy) * (
				8.0 * (u[i, j+1] - u[i, j-1]) - (u[i, j+2] - u[i, j-2])
			)
		adv_v_x[i, j] = u[i, j] / (12.0 * dx) * (
				8.0 * (v[i+1, j] - v[i-1, j]) - (v[i+2, j] - v[i-2, j])
			)
		adv_v_y[i, j] = v[i, j] / (12.0 * dy) * (
				8.0 * (v[i, j+1] - v[i, j-1]) - (v[i, j+2] - v[i, j-2])
			)

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _FifthOrder(BurgersAdvection):
	extent = 3

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		abs_u = gt.Equation()
		abs_v = gt.Equation()

		abs_u[i, j] = u[i, j] * (u[i, j] >= 0.0) - u[i, j] * (u[i, j] < 0)
		abs_v[i, j] = v[i, j] * (v[i, j] >= 0.0) - v[i, j] * (v[i, j] < 0)

		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (60.0 * dx) * (
				+ 45.0 * (u[i+1, j] - u[i-1, j])
				- 9.0 * (u[i+2, j] - u[i-2, j])
				+ (u[i+3, j] - u[i-3, j])
			) \
			- abs_u[i, j] / (60.0 * dx) * (
				+ (u[i+3, j] + u[i-3, j])
				- 6.0 * (u[i+2, j] + u[i-2, j])
				+ 15.0 * (u[i+1, j] + u[i-1, j])
				- 20.0 * u[i, j]
			)
		adv_u_y[i, j] = v[i, j] / (60.0 * dy) * (
				+ 45.0 * (u[i, j+1] - u[i, j-1])
				- 9.0 * (u[i, j+2] - u[i, j-2])
				+ (u[i, j+3] - u[i, j-3])
			) \
			- abs_v[i, j] / (60.0 * dy) * (
				+ (u[i, j+3] + u[i, j-3])
				- 6.0 * (u[i, j+2] + u[i, j-2])
				+ 15.0 * (u[i, j+1] + u[i, j-1])
				- 20.0 * u[i, j]
			)
		adv_v_x[i, j] = u[i, j] / (60.0 * dx) * (
				+ 45.0 * (v[i+1, j] - v[i-1, j])
				- 9.0 * (v[i+2, j] - v[i-2, j])
				+ (v[i+3, j] - v[i-3, j])
			) \
			- abs_u[i, j] / (60.0 * dx) * (
				+ (v[i+3, j] + v[i-3, j])
				- 6.0 * (v[i+2, j] + v[i-2, j])
				+ 15.0 * (v[i+1, j] + v[i-1, j])
				- 20.0 * v[i, j]
			)
		adv_v_y[i, j] = v[i, j] / (60.0 * dy) * (
				+ 45.0 * (v[i, j+1] - v[i, j-1])
				- 9.0 * (v[i, j+2] - v[i, j-2])
				+ (v[i, j+3] - v[i, j-3])
			) \
			- abs_v[i, j] / (60.0 * dy) * (
				+ (v[i, j+3] + v[i, j-3])
				- 6.0 * (v[i, j+2] + v[i, j-2])
				+ 15.0 * (v[i, j+1] + v[i, j-1])
				- 20.0 * v[i, j]
			)

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y


class _SixthOrder(BurgersAdvection):
	extent = 3

	@staticmethod
	def __call__(i, j, dx, dy, u, v):
		adv_u_x = gt.Equation()
		adv_u_y = gt.Equation()
		adv_v_x = gt.Equation()
		adv_v_y = gt.Equation()

		adv_u_x[i, j] = u[i, j] / (60.0 * dx) * (
				+ 45.0 * (u[i+1, j] - u[i-1, j])
				- 9.0 * (u[i+2, j] - u[i-2, j])
				+ (u[i+3, j] - u[i-3, j])
			)
		adv_u_y[i, j] = v[i, j] / (60.0 * dy) * (
				+ 45.0 * (u[i, j+1] - u[i, j-1])
				- 9.0 * (u[i, j+2] - u[i, j-2])
				+ (u[i, j+3] - u[i, j-3])
			)
		adv_v_x[i, j] = u[i, j] / (60.0 * dx) * (
				+ 45.0 * (v[i+1, j] - v[i-1, j])
				- 9.0 * (v[i+2, j] - v[i-2, j])
				+ (v[i+3, j] - v[i-3, j])
			)
		adv_v_y[i, j] = v[i, j] / (60.0 * dy) * (
				+ 45.0 * (v[i, j+1] - v[i, j-1])
				- 9.0 * (v[i, j+2] - v[i, j-2])
				+ (v[i, j+3] - v[i, j-3])
			)

		return adv_u_x, adv_u_y, adv_v_x, adv_v_y
