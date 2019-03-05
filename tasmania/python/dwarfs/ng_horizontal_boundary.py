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


class NGHorizontalBoundary:
	"""
	Abstract base class whose children handle the
	horizontal boundary conditions.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, nx, ny, nz, nb=None):
		"""
		Parameters
		----------
		nx : int
			Number of points featured by the *physical* grid
			along the first horizontal dimension.
		ny : int
			Number of points featured by the *physical* grid
			along the second horizontal dimension.
		nz : int
			Number of points featured by the *physical* grid
			along the vertical dimension.
		nb : `int`, optional
			Number of boundary layers.
		"""
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.nb = nb
		self.reference_state = None

	@property
	@abc.abstractmethod
	def mi(self):
		"""
		Return
		------
		int :
			Number of points featured by the *computational* grid
			along the first horizontal dimension.
		"""
		pass

	@property
	@abc.abstractmethod
	def mj(self):
		"""
		Return
		------
		int :
			Number of points featured by the *computational* grid
			along the second horizontal dimension.
		"""
		pass

	@abc.abstractmethod
	def get_computational_axis(self, paxis):
		"""
		Parameters
		----------
		paxis : dataarray_like
			1-D :class:`sympl.DataArray` representing a horizontal
			physical axis.

		Return
		------
		dataarray_like :
			1-D :class:`sympl.DataArray` representing the associated
			computational axis.
		"""
		pass

	@abc.abstractmethod
	def enforce(self, state):
		"""
		Enforce the horizontal boundary conditions on the passed state,
		which is modified in-place.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		"""
		pass

	@abc.abstractmethod
	def enforce_raw(self, state):
		"""
		Enforce the horizontal boundary conditions on the passed state,
		which is modified in-place.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting model state
			variables, and whose values are :class:`numpy.ndarray`\s
			storing values for those variables.
		"""
		pass


class Relaxed(NGHorizontalBoundary):
	"""
	Relaxed boundary conditions.
	"""