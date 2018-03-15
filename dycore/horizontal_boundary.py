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
Classes implementing horizontal boundary conditions.
"""
import abc
import numpy as np

from tasmania.namelist import datatype

class HorizontalBoundary:
	"""
	Abstract base class whose derived classes implement different types of horizontal boundary conditions.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	# Make the class abstract (credits: C. Zeman)
	__metaclass__ = abc.ABCMeta
	
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		self._grid = grid
		self.nb = nb

	@abc.abstractmethod
	def from_physical_to_computational_domain(self, phi):
		"""
		Given a :class:`numpy.ndarray` representing a physical field, return the associated stencils' computational
		domain, i.e., the :class:`numpy.ndarray` (accomodating the boundary conditions) which will be input 
		to the stencils. If the physical and computational fields coincide, a deep copy of the physical
		domain is returned.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the physical field.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the stencils' computational domain.

		Note
		----
		The implementation should be designed to work with both staggered and unstaggared fields.
		"""

	@abc.abstractmethod
	def from_computational_to_physical_domain(self, phi_, out_dims, change_sign):
		"""
		Given a :class:`numpy.ndarray` representing the computational domain of a stencil, return the 
		associated physicalm field which may (or may not) satisfy the horizontal boundary conditions. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi_ : array_like 
			:class:`numpy.ndarray` representing the computational domain of a stencil.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the field defined over the physical domain.
		"""

	@abc.abstractmethod
	def apply(self, phi_new, phi_now):
		"""
		Apply the boundary conditions on the field :attr:`phi_new`, possibly relying upon the solution 
		:attr:`phi_now` at the current time.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.
		"""

	@abc.abstractmethod
	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`x`- (i.e., :obj:`i`-) direction so to satisfy 
		the lateral boundary conditions. For this, possibly rely upon the field :obj:`phi_now` at the current time.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.
		"""

	@abc.abstractmethod
	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`y`- (i.e., :obj:`j`-) direction so to satisfy 
		the lateral boundary conditions. For this, possibly rely upon the field :obj:`phi_now` at the current time.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field at the current time.
		"""

	@staticmethod
	def factory(horizontal_boundary_type, grid, nb):
		"""
		Static method which returns an instance of the derived class which implements the boundary
		conditions specified by :data:`horizontal_boundary_type`.

		Parameters
		----------
		horizontal_boundary_type : str
			String specifying the type of boundary conditions to apply. Either:

				* 'periodic', for periodic boundary conditions;
				* 'relaxed', for relaxed boundary conditions;
				* 'relaxed-symmetric-xz', for relaxed boundary conditions for a :math:`xz`-symmetric field.
				* 'relaxed-symmetric-yz', for relaxed boundary conditions for a :math:`yz`-symmetric field.

		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.

		Return
		------
		obj :
			An instance of the derived class implementing the boundary conditions specified by 
			:data:`horizontal_boundary_type`.
		"""
		if horizontal_boundary_type == 'periodic':
			if grid.ny == 1:
				return PeriodicXZ(grid, nb)
			elif grid.nx == 1:
				return PeriodicYZ(grid, nb)
			else:
				return Periodic(grid, nb)
		elif horizontal_boundary_type == 'relaxed':
			if grid.ny == 1:
				return RelaxedXZ(grid, nb)
			elif grid.nx == 1:
				return RelaxedYZ(grid, nb)
			else:
				return Relaxed(grid, nb)
		elif horizontal_boundary_type == 'relaxed-symmetric-xz':
			return RelaxedSymmetricXZ(grid, nb)
		elif horizontal_boundary_type == 'relaxed-symmetric-yz':
			return RelaxedSymmetricYZ(grid, nb)
		else:
			raise ValueError('Unknown boundary conditions type.')

		
class Periodic(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally periodic boundary conditions.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2 * nb, nj + 2 * nb, nk), dtype = datatype)
		phi_[nb:-nb, nb:-nb, :] = phi

		# Extend in the x-direction
		phi_[0:nb, nb:-nb, :] = phi_[nx:nx+nb  , nb:-nb, :]
		phi_[-nb:, nb:-nb, :] = phi_[-nx-nb:-nx, nb:-nb, :]

		# Extend in the y-direction
		phi_[:, 0:nb, :] = phi_[:, ny:ny+nb  , :]
		phi_[:, -nb:, :] = phi_[:, -ny-nb:-ny, :]

		return phi_

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Shrink the field :obj:`phi_` by removing the :attr:`nb` outermost layers.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi_.shape

		# Check
		assert ((ni == nx + 2*nb) or (ni == nx + 1 + 2*nb)) and ((nj == ny + 2*nb) or (nj == ny + 1 + 2*nb)), \
			   'The input field does not have the dimensions one would expect.' \
			   'Hint: was the field extended?'

		# Shrink
		return phi_[nb:-nb, nb:-nb, :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, _ = phi_now.shape

		# Apply the periodic boundary conditions only if the field is staggered
		# If the field is unstaggered, then it should be automatically periodic 
		# once the computations are over, due to the periodic extension performed 
		# before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_x(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`x`-direction so to satisfy the periodic 
		boundary conditions. For this, the field :obj:`phi_now` at the current time is not required. Yet,
		it appears as (default) argument for compliancy with the general API.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		nx = self._grid.nx
		phi_new[0, :, :], phi_new[-1, :, :] = phi_new[nx-1, :, :], phi_new[-nx-1, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`y`-direction so to satisfy the periodic 
		boundary conditions. For this, the field :obj:`phi_now` at the current time is not required. Yet,
		it appears as (default) argument for compliancy with the general API.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		nx = self._grid.nx
		phi_new[:, 0, :], phi_new[:, -1, :] = phi_new[:, nx-1, :], phi_new[:, -nx-1, :]


class PeriodicXZ(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally periodic boundary conditions for fields defined
	on a computational domain consisting of only one grid point in the :math:`y`-direction.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers in the :obj:`j`-direction,
		then repeat it :data:`2 * nb + 1` times along the :obj:`i`-direction.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2 * nb, nj, nk), dtype = datatype)
		phi_[nb:-nb, :, :] = phi

		# Extend in the x-direction, then repeat in the y-direction
		phi_[0:nb, :, :] = phi_[nx:nx+nb  , :, :]
		phi_[-nb:, :, :] = phi_[-nx-nb:-nx, :, :]

		return np.concatenate((np.repeat(phi_[:, 0:1, :], nb, axis = 1),
							   phi_,
							   np.repeat(phi_[:, 0:1, :], nb, axis = 1)), axis = 1)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return the central :data:`i-k`-slice of the input field :data:`phi_`, removing the :attr:`nb` outermost 
		layers in :data:`i`-direction.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb, nj = self.nb, out_dims[1]
		return phi_[nb:-nb, nb:nb+nj, :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self._grid.nx, phi_now.shape[0]

		# Apply the periodic boundary conditions only if the field is staggered
		# If the field is unstaggered, then it should be automatically periodic 
		# once the computations are over, due to the periodic extension performed 
		# before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

	def set_outermost_layers_x(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`x`-direction so to satisfy the periodic 
		boundary conditions. For this, the field :obj:`phi_now` at the current time is not required. Yet,
		it appears as (default) argument for compliancy with the general API.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		nx = self._grid.nx
		phi_new[0, :, :], phi_new[-1, :, :] = phi_new[nx-1, :, :], phi_new[-nx-1, :, :]


class PeriodicYZ(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally periodic boundary conditions for fields 
	defined on a computational domain consisting of only one grid point in the :math:`x`-direction.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers in the :obj:`j`-direction,
		then repeat it :data:`2 * nb + 1` times along the :obj:`i`-direction.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((1, nj + 2 * nb, nk), dtype = datatype)
		phi_[:, nb:-nb, :] = phi

		# Extend in the x-direction, then repeat in the y-direction
		phi_[:, 0:nb, :] = phi_[:, nx:nx+nb  , :]
		phi_[:, -nb:, :] = phi_[:, -nx-nb:-nx, :]

		return np.repeat(phi_, 2 * nb + 1, axis = 0)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return the central :data:`j-k`-slice of the input field :data:`phi_`, removing the :attr:`nb` outermost 
		layers in :data:`j`-direction.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:nb+1, nb:-nb :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self._grid.ny, phi_now.shape[1]

		# Apply the periodic boundary conditions only if the field is staggered
		# If the field is unstaggered, then it should be automatically periodic 
		# once the computations are over, due to the periodic extension performed 
		# before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_y(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`y`-direction so to satisfy the periodic 
		boundary conditions. For this, the field :obj:`phi_now` at the current time is not required. Yet,
		it appears as (default) argument for compliancy with the general API.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		ny = self._grid.ny
		phi_new[:, 0, :], phi_new[:, -1, :] = phi_new[:, ny-1, :], phi_new[:, -ny-1, :]


class Relaxed(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally relaxed boundary conditions.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		# The relaxation coefficients
		self._rel = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype = datatype)
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices for a x-staggered field
		self._stgx_s = np.repeat(self._rel[np.newaxis, :] , grid.nx - 2 * self.nr + 1, axis = 0)
		self._stgx_n = np.repeat(self._rrel[np.newaxis, :], grid.nx - 2 * self.nr + 1, axis = 0)

		# The relaxation matrices for a y-staggered field
		self._stgy_w = np.repeat(self._rel[:, np.newaxis] , grid.ny - 2 * self.nr + 1, axis = 1)
		self._stgy_e = np.repeat(self._rrel[:, np.newaxis], grid.ny - 2 * self.nr + 1, axis = 1)

		# The corner relaxation matrices
		cnw = np.zeros((self.nr, self.nr), dtype = datatype)
		for i in range(self.nr):
			cnw[i, i:] = self._rel[i]
			cnw[i:, i] = self._rel[i]
		cne = cnw[:, ::-1]
		cse = np.transpose(cnw)
		csw = np.transpose(cne)

		# Append the corner relaxation matrices to their neighbours
		self._stgx_s = np.concatenate((cnw, self._stgx_s, cne), axis = 0)
		self._stgx_n = np.concatenate((csw, self._stgx_n, cse), axis = 0)
		self._stgy_w = np.concatenate((cnw, self._stgy_w, cne), axis = 1)
		self._stgy_e = np.concatenate((csw, self._stgy_e, cse), axis = 1)

		# Repeat all relaxation matrices along the z-axis
		self._stgx_s = np.repeat(self._stgx_s[:, :, np.newaxis], grid.nz + 1, axis = 2)
		self._stgx_n = np.repeat(self._stgx_n[:, :, np.newaxis], grid.nz + 1, axis = 2)
		self._stgy_w = np.repeat(self._stgy_w[:, :, np.newaxis], grid.nz + 1, axis = 2)
		self._stgy_e = np.repeat(self._stgy_e[:, :, np.newaxis], grid.nz + 1, axis = 2)

	def from_physical_to_computational_domain(self, phi):
		"""
		As no extension is required to apply relaxed boundary conditions, return a deep copy of the
		input field :obj:`phi`.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A deep copy of :obj:`phi`.
		"""
		return np.copy(phi)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		As no extension is required to apply relaxed boundary conditions, return a deep copy of the
		input field :obj:`phi_`.

		Parameters
		----------
		phi_ : array_like 
			A :class:`numpy.ndarray`.
		out_dims : `tuple`, optional
			Tuple of the output array dimensions.
		change_sign : `bool`, optional
			:obj:`True` if the field should change sign through the symmetry plane, :obj:`False` otherwise.

		Return
		------
		array_like :
			A deep copy of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		return np.copy(phi_)
	
	def apply(self, phi_new, phi_now):
		"""
		Apply relaxed lateral boundary conditions. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		# Shortcuts
		nx, ny, nz, nb, nr = self._grid.nx, self._grid.ny, self._grid.nz, self.nb, self.nr
		ni, nj, nk = phi_now.shape

		# The boundary values
		west  = np.repeat(phi_now[0:1,   :, :], nr, axis = 0)
		east  = np.repeat(phi_now[-1:,   :, :], nr, axis = 0)
		south = np.repeat(phi_now[  :, 0:1, :], nr, axis = 1)
		north = np.repeat(phi_now[  :, -1:, :], nr, axis = 1)

		# Apply the relaxed boundary conditions in the x-direction
		if nj == ny: # unstaggered
			phi_new[ :nr, :, :] = self._stgy_w[:, :-1, :nk] * west + (1 - self._stgy_w[:, :-1, :nk]) * phi_now[ :nr, :, :]
			phi_new[-nr:, :, :] = self._stgy_e[:, :-1, :nk] * east + (1 - self._stgy_e[:, :-1, :nk]) * phi_now[-nr:, :, :]
		else:		 # y-staggered
			phi_new[ :nr, :, :] = self._stgy_w[:, :, nk] * west + (1 - self._stgy_w[:, :, :nk]) * phi_now[ :nr, :, :]
			phi_new[-nr:, :, :] = self._stgy_e[:, :, nk] * east + (1 - self._stgy_e[:, :, :nk]) * phi_now[-nr:, :, :]

		# Apply the relaxed boundary conditions in the y-direction
		if ni == nx: # unstaggered
			phi_new[:,  :nr, :] = self._stgx_s[:-1, :, :nk] * south + (1 - self._stgx_s[:-1, :, :nk]) * phi_now[:,  :nr, :]
			phi_new[:, -nr:, :] = self._stgx_n[:-1, :, :nk] * north + (1 - self._stgx_n[:-1, :, :nk]) * phi_now[:, -nr:, :]
		else:		 # x-staggered
			phi_new[:,  :nr, :] = self._stgx_s[:, :, nk] * south + (1 - self._stgx_s[:, :, nk]) * phi_now[:,  :nr, :]
			phi_new[:, -nr:, :] = self._stgx_n[:, :, nk] * north + (1 - self._stgx_n[:, :, nk]) * phi_now[:, -nr:, :]

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`x`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :obj:`x`-direction.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		phi_new[0, :, :], phi_new[-1, :, :] = phi_now[0, :, :], phi_now[-1, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`y`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichelt conditions in :obj:`y`-direction.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		phi_new[:, 0, :], phi_new[:, -1, :] = phi_now[:, 0, :], phi_now[:, -1, :]


class RelaxedXZ(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally relaxed boundary conditions for fields defined
	on a computational domain consisting of only one grid point in the :math:`y`-direction.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		# The relaxation coefficients
		self._rel = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype = datatype)
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices
		self._stg_w = np.repeat(self._rel[:, np.newaxis, np.newaxis] , grid.nz + 1, axis = 2)
		self._stg_e = np.repeat(self._rrel[:, np.newaxis, np.newaxis], grid.nz + 1, axis = 2)

	def from_physical_to_computational_domain(self, phi):
		"""
		While no extension is required to apply relaxed boundary conditions along the :math:`x`-direction, 
		the field gets repeated :data:`2 * nb` times along the :math:`y`-direction, i.e., the second array axis.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A deep copy of :obj:`phi`, repeated :data:`2 * nb + 1` times along the second array axis.
		"""
		nb = self.nb
		phi_ = np.copy(phi)
		return np.concatenate((np.repeat(phi_[:, 0:1, :], nb, axis = 1),
							   phi_,
							   np.repeat(phi_[:, 0:1, :], nb, axis = 1)), axis = 1)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Remove the :data:`2 * nb` outermost :data:`i-k`-slices from the input field :obj:`phi_`.

		Parameters
		----------
		phi_ : array_like 
			A :class:`numpy.ndarray`.
		out_dims : `tuple`, optional
			Tuple of the output array dimensions.
		change_sign : `bool`, optional
			:obj:`True` if the field should change sign through the symmetry plane, :obj:`False` otherwise.

		Return
		------
		array_like :
			A deep copy of the central :data:`i-k`-slice(s) of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return np.copy(phi_[:, nb:-nb, :])
	
	def apply(self, phi_new, phi_now):
		"""
		Apply relaxed lateral boundary conditions along the :math:`x`-axis. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		# Shortcuts
		nx, nz, nb, nr = self._grid.nx, self._grid.nz, self.nb, self.nr
		ni, _, nk = phi_now.shape

		# The boundary values
		west  = np.repeat(phi_now[0:1, :, :], nr, axis = 0)
		east  = np.repeat(phi_now[-1:, :, :], nr, axis = 0)

		# Apply the relaxed boundary conditions in the x-direction
		phi_new[ :nr, :, :] = self._stg_w[:, :, :nk] * west + (1 - self._stg_w[:, :, :nk]) * phi_now[ :nr, :, :]
		phi_new[-nr:, :, :] = self._stg_e[:, :, :nk] * east + (1 - self._stg_e[:, :, :nk]) * phi_now[-nr:, :, :]

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`x`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :obj:`x`-direction.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		phi_new[0, :, :], phi_new[-1, :, :] = phi_now[0, :, :], phi_now[-1, :, :]


class RelaxedYZ(HorizontalBoundary):
	"""
	This class inherits :class:`HorizontalBoundary` to implement horizontally relaxed boundary conditions for fields defined
	on a computational domain consisting of only one grid point in the :math:`x`-direction.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		# The relaxation coefficients
		self._rel = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype = datatype)
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices
		self._stg_s = np.repeat(self._rel[np.newaxis, :, np.newaxis] , grid.nz, axis = 2)
		self._stg_n = np.repeat(self._rrel[np.newaxis, :, np.newaxis], grid.nz, axis = 2)

	def from_physical_to_computational_domain(self, phi):
		"""
		While no extension is required to apply relaxed boundary conditions along the :math:`y`-direction, 
		the field gets repeated :data:`2 * nb + 1` times along the :math:`x`-direction, i.e., the first array axis.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A deep copy of :obj:`phi`, repeated :data:`2 * nb + 1` times along the first array axis.
		"""
		return np.repeat(phi, 2 * self.nb + 1, axis = 0)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return a deep copy of the central :data:`j-k`-slice of the input field :obj:`phi_`.

		Parameters
		----------
		phi_ : array_like 
			A :class:`numpy.ndarray`.
		out_dims : `tuple`, optional
			Tuple of the output array dimensions.
		change_sign : `bool`, optional
			:obj:`True` if the field should change sign through the symmetry plane, :obj:`False` otherwise.

		Return
		------
		array_like :
			A deep copy of the central :data:`j-k`-slice of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		return np.copy(phi_[self.nb:self.nb + 1, :, :])
	
	def apply(self, phi_new, phi_now):
		"""
		Apply relaxed lateral boundary conditions along the :math:`x`-axis. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		# Shortcuts
		ny, nz, nb, nr = self._grid.ny, self._grid.nz, self.nb, self.nr
		_, nj, _ = phi_now.shape

		# The boundary values
		south = np.repeat(phi_now[:, 0:1, :], nr, axis = 1)
		north = np.repeat(phi_now[:, -1:, :], nr, axis = 1)

		# Apply the relaxed boundary conditions in the x-direction
		phi_new[:,  :nr, :] = self._stg_s * south + (1 - self._stg_s) * phi_now[:,  :nr, :]
		phi_new[:, -nr:, :] = self._stg_n * north + (1 - self._stg_n) * phi_now[:, -nr:, :]

	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :obj:`y`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :obj:`y`-direction.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : array_like 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be time-independent, so that they 
		can be inferred from the solution at current time.
		"""
		phi_new[:, 0, :], phi_new[:, -1, :] = phi_now[:, 0, :], phi_now[:, -1, :]


class RelaxedSymmetricXZ(Relaxed):
	"""
	This class inherits :class:`Relaxed` to implement horizontally relaxed boundary conditions
	for fields symmetric with respect to the :math:`xz`-plane :math:`y = y_c = 0.5 (a_y + b_y)`,
	where :math:`a_y` and :math:`b_y` denote the extremes of the domain in the :math:`y`-direction.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	nr : int 
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Return the :math:`y`-lowermost half of the domain. To accomodate symmetric conditions,
		we retain (at least) :attr:`nb` additional layers in the positive direction of the :math:`y`-axis.

		Parameters
		----------
		phi : array_like 
			:class:`numpy.ndarray` representing the physical field.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the stencils' computational domain.
		"""
		ny, nj, nb = self._grid.ny, phi.shape[1], self.nb
		half = int((nj + 1) / 2)

		if nj % 2 == 0 and nj == ny + 1: 
			return np.copy(phi[:, :half + nb + 1, :])
		return np.copy(phi[:, :half + nb, :])

	def from_computational_to_physical_domain(self, phi_, out_dims, change_sign = False):
		"""
		Mirror the computational domain with respect to the :math:`xz`-plane :math:`y = y_c`.

		Parameters
		----------
		phi_ : array_like 
			:class:`numpy.ndarray` representing the computational domain of a stencil.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : `bool`, optional 
			:obj:`True` if the field should change sign through the symmetry plane, :obj:`False` otherwise.
			Default is false.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the field defined over the physical domain.
		"""
		nb = self.nb
		half = phi_.shape[1] - nb

		phi = np.zeros(out_dims, dtype = datatype)
		phi[:, :half, :] = phi_[:, :half, :]

		if out_dims[1] % 2 == 0:
			phi[:, half:, :] = - np.flip(phi[:, :half, :], axis = 1) if change_sign else np.flip(phi[:, :half, :], axis = 1)
		else:
			phi[:, half:, :] = - np.flip(phi[:, :half-1, :], axis = 1) if change_sign else np.flip(phi[:, :half-1, :], axis = 1)

		return phi


class RelaxedSymmetricYZ(Relaxed):
	"""
	This class inherits :class:`Relaxed` to implement horizontally relaxed boundary conditions
	for fields symmetric with respect to the :math:`yz`-plane :math:`x = x_c = 0.5 (a_x + b_x)`,
	where :math:`a_x` and :math:`b_x` denote the extremes of the domain in the :math:`x`-direction.

	Attributes
	----------
	nb : int 
		Number of boundary layers.
	nr : int 
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Return the :math:`x`-lowermost half of the domain. To accomodate symmetric conditions,
		we retain (at least) :attr:`nb` additional layers in the positive direction of the :math:`x`-axis.

		Parameters
		----------
		phi : array_like
			:class:`numpy.ndarray` representing the physical field.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the stencils' computational domain.
		"""
		nx, ni, nb = self._grid.nx, phi.shape[0], self.nb
		half = int((ni + 1) / 2)

		if ni % 2 == 0 and ni == nx + 1: 
			return np.copy(phi[:half + nb + 1, :, :])
		return np.copy(phi[:half + nb, :, :])

	def from_computational_to_physical_domain(self, phi_, out_dims, change_sign = False):
		"""
		Mirror the computational domain with respect to the :math:`yz`-axis :math:`x = x_c`.

		Parameters
		----------
		phi_ : array_like 
			:class:`numpy.ndarray` representing the computational domain of a stencil.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : `bool`, optional 
			:obj:`True` if the field should change sign through the symmetry plane, :obj:`False` otherwise.
			Default is false.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the field defined over the physical domain.
		"""
		nb = self.nb
		half = phi_.shape[0] - nb

		phi = np.zeros(out_dims, dtype = datatype)
		phi[:half, :, :] = phi_[:half, :, :]

		if out_dims[0] % 2 == 0:
			phi[half:, :, :] = - np.flip(phi[:half, :, :], axis = 0) if change_sign else np.flip(phi[:half, :, :], axis = 0)
		else:
			phi[half:, :, :] = - np.flip(phi[:half-1, :, :], axis = 0) if change_sign else np.flip(phi[:half-1, :, :], axis = 0)

		return phi
