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
	HorizontalBoundary
"""
import abc
import numpy as np

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class HorizontalBoundary:
	"""
	Abstract base class whose derived classes implement different
	types of horizontal boundary conditions.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta
	
	def __init__(self, grid, nb, init_time=None):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		init_time : `datetime`, optional
			TODO
		"""
		self._grid, self._nb = grid, nb
		self._init_time = init_time

	@property
	def nb(self):
		"""
		Return
		------
		int :
			Return the number of boundary layers.

		Raises
		------
		ValueError :
			If the number of boundary layers has not been set.
		"""
		if self._nb is None:
			raise ValueError('Number of boundary layers not set.')
		else:
			return self._nb

	@nb.setter
	def nb(self, val):
		"""
		Parameters
		----------
		val : int
			Number of boundary layers.
		"""
		self._nb = val

	@property
	@abc.abstractmethod
	def mi(self):
		"""
		Get the extent of the computational domain in the :math:`x`-direction.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Return
		------
		int :
			The extent of the computational domain in the :math:`x`-direction.
		"""

	@property
	@abc.abstractmethod
	def mj(self):
		"""
		Get the extent of the computational domain in the :math:`y`-direction.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Return
		------
		int :
			The extent of the computational domain in the :math:`y`-direction.
		"""

	@abc.abstractmethod
	def from_physical_to_computational_domain(self, phi):
		"""
		Given a :class:`numpy.ndarray` representing a physical field,
		return the associated stencil's computational domain, i.e., the
		:class:`numpy.ndarray` (accommodating the boundary conditions)
		which will be input to the stencil. If the physical and computational
		fields coincide, a shallow copy of the physical domain is returned.
		As this method is marked as abstract, its implementation is delegated
		to the derived classes.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the physical field.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the stencils'
			computational domain.

		Note
		----
		The implementation should be designed to work with both
		staggered and unstaggered fields.
		"""

	@abc.abstractmethod
	def from_computational_to_physical_domain(
		self, phi_, out_dims, change_sign
	):
		"""
		Given a :class:`numpy.ndarray` representing the computational
		domain of a stencil, return the associated physical field which
		may (or may not) satisfy the horizontal boundary conditions.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		phi_ : array_like 
			:class:`numpy.ndarray` representing the computational domain
			of a stencil.
		out_dims : tuple 
			The shape of the output array.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry
			plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the physical field.
		"""

	@abc.abstractmethod
	def enforce(self, phi_new, phi_now, field_name, time):
		"""
		Enforce the boundary conditions on the field :attr:`phi_new`,
		possibly relying upon the solution :attr:`phi_now` at the
		current time.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which
			applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.
		field_name : str
			TODO
		time : datetime
			TODO
		"""

	@abc.abstractmethod
	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`x`-direction so to satisfy the lateral boundary
		conditions. For this, possibly rely upon the field
		:obj:`phi_now` at the current time.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which
			applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the
			current time.
		"""

	@abc.abstractmethod
	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`y`-direction so to satisfy the lateral boundary
		conditions. For this, possibly rely upon the field
		:obj:`phi_now` at the current time.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which
			applying the boundary conditions.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field at the
			current time.
		"""
		
	@abc.abstractmethod
	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Return
		------
		obj :
			Instance of the same class of
			:obj:`tasmania.dynamics.horizontal_boundary.HorizontalBoundary._grid`
			representing the underlying computational grid.
		"""

	@staticmethod
	def factory(horizontal_boundary_type, grid, nb=None, init_time=None, **kwargs):
		"""
		Static method which returns an instance of the derived
		class which implements the boundary conditions specified by
		:data:`horizontal_boundary_type`.

		Parameters
		----------
		horizontal_boundary_type : str
			String specifying the type of boundary conditions to apply. Either:

				* 'periodic', for periodic boundary conditions;
				* 'relaxed', for relaxed boundary conditions;
				* 'relaxed_symmetric_xz', for relaxed boundary conditions \
					for a :math:`xz`-symmetric field.
				* 'relaxed_symmetric_yz', for relaxed boundary conditions \
					for a :math:`yz`-symmetric field.

		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : `int`, optional
			Number of boundary layers.
		init_time : `datetime`, optional
			TODO

		Return
		------
		obj :
			An instance of the derived class implementing the
			boundary conditions specified by :data:`horizontal_boundary_type`.
		"""
		import spike._horizontal_boundary as module

		if horizontal_boundary_type == 'periodic':
			if grid.ny == 1:
				return module._PeriodicXZ(grid, nb)
			elif grid.nx == 1:
				return module._PeriodicYZ(grid, nb)
			else:
				return module._Periodic(grid, nb)
		
		if horizontal_boundary_type == 'relaxed':
			if grid.ny == 1:
				return module._RelaxedXZ(grid, nb)
			elif grid.nx == 1:
				return module._RelaxedYZ(grid, nb)
			else:
				return module._Relaxed(grid, nb)

		if horizontal_boundary_type == 'relaxed_symmetric_xz':
			return module._RelaxedSymmetricXZ(grid, nb)

		if horizontal_boundary_type == 'relaxed_symmetric_yz':
			return module._RelaxedSymmetricYZ(grid, nb)

		import spike.burgers_horizontal_boundary as module
		if horizontal_boundary_type == 'zhao':
			return module.ZhaoHorizontalBoundary(grid, nb, init_time, **kwargs)
		
		raise ValueError('Unknown boundary conditions type.')
