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
Classes implementing relaxed horizontal boundary conditions.
"""
import abc
import copy
import numpy as np

from tasmania.dycore.horizontal_boundary import HorizontalBoundary
from tasmania.namelist import datatype

class Relaxed(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	relaxed boundary conditions.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
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
		As no extension is required to apply relaxed boundary conditions, return a shallow copy of the
		input field :obj:`phi`.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A shallow copy of :obj:`phi`.
		"""
		return phi

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		As no extension is required to apply relaxed boundary conditions, return a shallow copy of the
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
			A shallow copy of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		return phi_
	
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
		nx, ny, nz, nb, nr = self.grid.nx, self.grid.ny, self.grid.nz, self.nb, self.nr
		ni, nj, nk = phi_now.shape

		# The boundary values
		west  = np.repeat(phi_now[0:1,   :, :], nr, axis = 0)
		east  = np.repeat(phi_now[-1:,   :, :], nr, axis = 0)
		south = np.repeat(phi_now[  :, 0:1, :], nr, axis = 1)
		north = np.repeat(phi_now[  :, -1:, :], nr, axis = 1)

		# Set the outermost layers
		phi_new[ :nb, nb:-nb, :] = phi_now[ :nb, nb:-nb, :]
		phi_new[-nb:, nb:-nb, :] = phi_now[-nb:, nb:-nb, :]
		phi_new[:,  :nb, :] = phi_now[:,  :nb, :]
		phi_new[:, -nb:, :] = phi_now[:, -nb:, :]

		# Apply the relaxed boundary conditions in the x-direction
		if nj == ny: # unstaggered
			phi_new[ :nr, :, :] = self._stgy_w[:, :-1, :nk] * west + (1. - self._stgy_w[:, :-1, :nk]) * phi_new[ :nr, :, :]
			phi_new[-nr:, :, :] = self._stgy_e[:, :-1, :nk] * east + (1. - self._stgy_e[:, :-1, :nk]) * phi_new[-nr:, :, :]
		else:		 # y-staggered
			phi_new[ :nr, :, :] = self._stgy_w[:, :, nk] * west + (1. - self._stgy_w[:, :, :nk]) * phi_new[ :nr, :, :]
			phi_new[-nr:, :, :] = self._stgy_e[:, :, nk] * east + (1. - self._stgy_e[:, :, :nk]) * phi_new[-nr:, :, :]

		# Apply the relaxed boundary conditions in the y-direction
		if ni == nx: # unstaggered
			phi_new[:,  :nr, :] = self._stgx_s[:-1, :, :nk] * south + (1. - self._stgx_s[:-1, :, :nk]) * phi_new[:,  :nr, :]
			phi_new[:, -nr:, :] = self._stgx_n[:-1, :, :nk] * north + (1. - self._stgx_n[:-1, :, :nk]) * phi_new[:, -nr:, :]
		else:		 # x-staggered
			phi_new[:,  :nr, :] = self._stgx_s[:, :, nk] * south + (1. - self._stgx_s[:, :, nk]) * phi_new[:,  :nr, :]
			phi_new[:, -nr:, :] = self._stgx_n[:, :, nk] * north + (1. - self._stgx_n[:, :, nk]) * phi_new[:, -nr:, :]

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`x`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :math:`x`-direction.

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
		Set the outermost layers of :obj:`phi_new` in the :math:`y`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichelt conditions in :math:`y`-direction.

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

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		return copy.deepcopy(self.grid)


class RelaxedXZ(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	relaxed boundary conditions for fields defined on a computational domain consisting of only one grid point 
	in the :math:`y`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
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
		:data:`nb` ghost layers are appended in the :math:`y`-direction.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A deep copy of :obj:`phi`, with :data:`nb` ghost layers in the :math:`y`-direction.
		"""
		nb = self.nb
		return np.concatenate((np.repeat(phi[:, 0:1, :], nb, axis = 1),
							   phi,
							   np.repeat(phi[:, -1:, :], nb, axis = 1)), axis = 1)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Remove the :data:`nb` outermost :math:`xz`-slices from the input field :obj:`phi_`.

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
			The central :math:`xz`-slice(s) of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[:, nb:-nb, :]
	
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
		nx, nz, nb, nr = self.grid.nx, self.grid.nz, self.nb, self.nr
		ni, _, nk = phi_now.shape

		# The boundary values
		west  = np.repeat(phi_now[0:1, :, :], nr, axis = 0)
		east  = np.repeat(phi_now[-1:, :, :], nr, axis = 0)

		# Apply the relaxed boundary conditions in the x-direction
		phi_new[ :nr, :, :] = self._stg_w[:, :, :nk] * west + (1. - self._stg_w[:, :, :nk]) * phi_new[ :nr, :, :]
		phi_new[-nr:, :, :] = self._stg_e[:, :, :nk] * east + (1. - self._stg_e[:, :, :nk]) * phi_new[-nr:, :, :]

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`x`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :math:`x`-direction.

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

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0], x[-1]]
		_nx       = nx
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0] - nb * dy, y[-1] + nb * dy]
		_ny       = ny + 2 * nb
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)


class RelaxedYZ(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	relaxed boundary conditions for fields defined on a computational domain consisting of only one grid point 
	in the :math:`x`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
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
		:data:`nb` ghost layers are appended in the :math:`x`-direction.

		Parameters
		----------
		phi : array_like 
			A :class:`numpy.ndarray`.

		Return
		------
		array_like :
			A deep copy of :obj:`phi`, with :data:`nb` ghost layers along the :math:`y`-layers.
		"""
		nb = self.nb
		return np.concatenate((np.repeat(phi[0:1, :, :], nb, axis = 0),
							   phi,
							   np.repeat(phi[-1:, :, :], nb, axis = 0)), axis = 0)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return a deep copy of the central :math:`yz`-slices of the input field :obj:`phi_`.

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
			The central :math:`yz`-slice(s) of :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		return phi_[self.nb:-self.nb, :, :]
	
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
		ny, nz, nb, nr = self.grid.ny, self.grid.nz, self.nb, self.nr
		_, nj, _ = phi_now.shape

		# The boundary values
		south = np.repeat(phi_now[:, 0:1, :], nr, axis = 1)
		north = np.repeat(phi_now[:, -1:, :], nr, axis = 1)

		# Apply the relaxed boundary conditions in the x-direction
		phi_new[:,  :nr, :] = self._stg_s * south + (1. - self._stg_s) * phi_new[:,  :nr, :]
		phi_new[:, -nr:, :] = self._stg_n * north + (1. - self._stg_n) * phi_new[:, -nr:, :]

	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`y`-direction equal to the corresponding 
		layers of :obj:`phi_now`. In other words, apply Dirichlet conditions in :math:`y`-direction.

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

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0] - nb * dx, x[-1] + nb * dx]
		_nx       = nx + 2 * nb
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0], y[-1]]
		_ny       = ny
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)


class RelaxedSymmetricXZ(Relaxed):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary_relaxed.Relaxed` to implement horizontally 
	relaxed boundary conditions for fields symmetric with respect to the :math:`xz`-plane :math:`y = y_c = 0.5 (a_y + b_y)`,
	where :math:`a_y` and :math:`b_y` denote the extremes of the domain in the :math:`y`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
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
		ny, nj, nb = self.grid.ny, phi.shape[1], self.nb
		half = int((nj + 1) / 2)

		if nj % 2 == 0 and nj == ny + 1: 
			return phi[:, :half + nb + 1, :]
		return phi[:, :half + nb, :]

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
			phi[:, half:, :] = - np.flip(phi[:, :half, :], axis = 1) if change_sign else \
							   np.flip(phi[:, :half, :], axis = 1)
		else:
			phi[:, half:, :] = - np.flip(phi[:, :half-1, :], axis = 1) if change_sign else \
							   np.flip(phi[:, :half-1, :], axis = 1)

		return phi

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0], x[-1]]
		_nx       = nx
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		half = int((ny - 1) / 2)
		_domain_y = [y[0], y[half + nb]]
		_ny       = half + nb + 1
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)


class RelaxedSymmetricYZ(Relaxed):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary_relaxed.Relaxed` to implement horizontally 
	relaxed boundary conditions for fields symmetric with respect to the :math:`yz`-plane :math:`x = x_c = 0.5 (a_x + b_x)`,
	where :math:`a_x` and :math:`b_x` denote the extremes of the domain in the :math:`x`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
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
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
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
		nx, ni, nb = self.grid.nx, phi.shape[0], self.nb
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

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		half = int((nx - 1) / 2)
		_domain_x = [x[0], x[half + nb]]
		_nx       = half + nb + 1
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0], y[-1]]
		_ny       = ny
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)
