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
	HorizontalVelocity
	WaterConstituent
"""
import numpy as np

import gridtools as gt

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class HorizontalVelocity:
	"""
	This class diagnoses the horizontal momenta (respectively, velocity
	components) with the help of the air density and the horizontal
	velocity components (resp., momenta).
	"""
	def __init__(self, grid, staggering=True, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		staggering : `bool`, optional
			:obj:`True` if the velocity components should be computed
			on the staggered grid, :obj:`False` to collocate the velocity
			components in the mass points.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		"""
		# Store the input arguments
		self._grid    = grid
		self._stag    = staggering
		self._backend = backend
		self._dtype   = dtype

		# Initialize the underlying stencils
		self._stencil_diagnosing_momenta_initialize()
		self._stencil_diagnosing_velocity_x_initialize()
		self._stencil_diagnosing_velocity_y_initialize()

	def get_momenta(self, d, u, v, du, dv):
		"""
		Diagnose the horizontal momenta.

		Parameters
		----------
		d : class.ndarray
			The air density.
		u : class.ndarray
			The x-velocity field.
		v : class.ndarray
			The y-velocity field.
		du : class.ndarray
			The buffer where the x-momentum will be written.
		dv : class.ndarray
			The buffer where the y-momentum will be written.
		"""
		# Update the arrays which serve as stencil's inputs
		self._d[...] = d[...]
		self._u[...] = u[...]
		self._v[...] = v[...]

		# Call the stencil's compute function
		self._stencil_diagnosing_momenta.compute()

		# Write the output into the provided buffers
		du[...] = self._du[...]
		dv[...] = self._dv[...]

	def get_velocity_components(self, d, du, dv, u, v):
		"""
		Diagnose the horizontal velocity components.

		Parameters
		----------
		d : class.ndarray
			The air density.
		du : class.ndarray
			The x-momentum.
		dv : class.ndarray
			The y-momentum.
		u : class.ndarray
			The buffer where the x-velocity will be written.
		v : class.ndarray
			The buffer where the y-velocity will be written.

		Note
		----
		If staggering is enabled, the first and last rows (respectively, columns)
		of the x-velocity (resp., y-velocity) are not set.
		"""
		# Update the arrays which serve as stencils' inputs
		self._d[...]  = d[...]
		self._du[...] = du[...]
		self._dv[...] = dv[...]

		# Call the stencils' compute function
		self._stencil_diagnosing_velocity_x.compute()
		self._stencil_diagnosing_velocity_y.compute()

		# Write the output into the provided buffers
		u[...] = self._u[...]
		v[...] = self._v[...]

	def _stencil_diagnosing_momenta_initialize(self):
		"""
		Initialize the GT4Py stencil diagnosing the horizontal momenta.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi = nx+1 if self._stag else nx
		mj = ny+1 if self._stag else ny

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_u'):
			self._u = np.zeros((mi, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_v'):
			self._v = np.zeros((nx, mj, nz), dtype=self._dtype)

		# Allocate the Numpy arrays which will serve as stencil's outputs
		if not hasattr(self, '_du'):
			self._du = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_dv'):
			self._dv = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_momenta = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_momenta_defs,
			inputs={'in_d': self._d, 'in_u': self._u, 'in_v': self._v},
			outputs={'out_du': self._du, 'out_dv': self._dv},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	def _stencil_diagnosing_momenta_defs(self, in_d, in_u, in_v):
		"""
		GT4Py stencil diagnosing the horizontal momenta.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_u : gridtools.Equation
			The x-velocity.
		in_v : gridtools.Equation
			The y-velocity.

		Returns
		-------
		out_du : gridtools.Equation
			The x-momentum.
		out_dv : gridtools.Equation
			The y-momentum.
		"""
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output fields
		out_du = gt.Equation()
		out_dv = gt.Equation()

		# Computations
		if self._stag:
			out_du[i, j] = 0.5 * in_d[i, j] * (in_u[i, j] + in_u[i+1, j])
			out_dv[i, j] = 0.5 * in_d[i, j] * (in_v[i, j] + in_v[i, j+1])
		else:
			out_du[i, j] = in_d[i, j] * in_u[i, j]
			out_dv[i, j] = in_d[i, j] * in_v[i, j]

		return out_du, out_dv

	def _stencil_diagnosing_velocity_x_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the x-velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi = nx+1 if self._stag else nx

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_du'):
			self._du = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's outputs
		if not hasattr(self, '_u'):
			self._u = np.zeros((mi, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		xstart = 1 if self._stag else 0
		self._stencil_diagnosing_velocity_x = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_velocity_x_defs,
			inputs={'in_d': self._d, 'in_du': self._du},
			outputs={'out_u': self._u},
			domain=gt.domain.Rectangle((xstart, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	def _stencil_diagnosing_velocity_y_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the y-velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mj = ny+1 if self._stag else ny

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_dv'):
			self._dv = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's outputs
		if not hasattr(self, '_v'):
			self._v = np.zeros((nx, mj, nz), dtype=self._dtype)

		# Instantiate the stencil
		ystart = 1 if self._stag else 0
		self._stencil_diagnosing_velocity_y = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_velocity_y_defs,
			inputs={'in_d': self._d, 'in_dv': self._dv},
			outputs={'out_v': self._v},
			domain=gt.domain.Rectangle((0, ystart, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	def _stencil_diagnosing_velocity_x_defs(self, in_d, in_du):
		"""
		GT4Py stencil diagnosing the x-component of the velocity.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_du : gridtools.Equation
			The x-momentum.

		Returns
		-------
		gridtools.Equation :
			The diagnosed x-velocity.
		"""
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_u = gt.Equation()

		# Computations
		if self._stag:
			out_u[i, j] = (in_du[i-1, j] + in_du[i, j]) / (in_d[i-1, j] + in_d[i, j])
		else:
			out_u[i, j] = in_du[i, j] / in_d[i, j]

		return out_u

	def _stencil_diagnosing_velocity_y_defs(self, in_d, in_dv):
		"""
		GT4Py stencil diagnosing the y-component of the velocity.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_dv : gridtools.Equation
			The y-momentum.

		Returns
		-------
		gridtools.Equation :
			The diagnosed y-velocity.
		"""
		# Indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Output field
		out_v = gt.Equation()

		# Computations
		if self._stag:
			out_v[i, j] = (in_dv[i, j-1] + in_dv[i, j]) / (in_d[i, j-1] + in_d[i, j])
		else:
			out_v[i, j] = in_dv[i, j] / in_d[i, j]

		return out_v


class WaterConstituent:
	"""
	This class diagnoses the density (respectively, mass fraction) of any water
	constituent with the help of the air density and the mass fraction (resp.,
	the density) of that water constituent.
	"""
	def __init__(self, grid, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		"""
		# Keep track of input arguments
		self._grid    = grid
		self._backend = backend
		self._dtype   = dtype

		# Initialize the pointers to the underlying stencils
		# These will be properly re-directed the first time the corresponding
		# entry-point method is invoked
		self._stencil_diagnosing_density = None
		self._stencil_diagnosing_and_clipping_density = None
		self._stencil_diagnosing_mass_fraction = None
		self._stencil_diagnosing_and_clipping_mass_fraction = None

	def get_density_of_water_constituent(self, d, q, dq, clipping=False):
		"""
		Diagnose the density of a water constituent.

		Parameters
		----------
		d : numpy.ndarray
			The air density.
		q : numpy.ndarray
			The mass fraction of the water constituent, in units of [g g^-1].
		dq : numpy.ndarray
			Buffer which will store the output density of the water constituent,
			in the same units of the input air density.
		clipping : `bool`, optional
			:obj:`True` to clip the negative values of the output field,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		"""
		# Initialize the underlying GT4Py stencils
		if self._stencil_diagnosing_density is None:
			self._stencil_diagnosing_density_initialize()

		# Update the arrays which serve as stencil's inputs
		self._d[...] = d[...]
		self._q[...] = q[...]

		# Set pointer to correct stencil
		stencil = self._stencil_diagnosing_and_clipping_density if clipping \
			else self._stencil_diagnosing_density

		# Run the stencil's compute function
		stencil.compute()

		# Set the output array
		dq[...] = self._dq[...]

	def get_mass_fraction_of_water_constituent_in_air(self, d, dq, q, clipping=False):
		"""
		Diagnose the mass fraction of a water constituent.

		Parameters
		----------
		d : numpy.ndarray
			The air density.
		dq : numpy.ndarray
			The density of the water constituent, in the same units of the input
			air density.
		q : numpy.ndarray
			Buffer which will store the output mass fraction of the water constituent,
			in the same units of the input air density.
		clipping : `bool`, optional
			:obj:`True` to clip the negative values of the output field,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		"""
		# Initialize the underlying GT4Py stencils
		if self._stencil_diagnosing_mass_fraction is None:
			self._stencil_diagnosing_mass_fraction_initialize()

		# Update the arrays which serve as stencil's inputs
		self._d[...]  = d[...]
		self._dq[...] = dq[...]

		# Set pointer to correct stencil
		stencil = self._stencil_diagnosing_and_clipping_mass_fraction if clipping \
			else self._stencil_diagnosing_mass_fraction

		# Run the stencil's compute function
		stencil.compute()

		# Update the output array
		q[...] = self._q[...]

	def _stencil_diagnosing_density_initialize(self):
		"""
		Initialize the GT4Py stencils in charge of diagnosing the density
		of the water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_q'):
			self._q = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_dq'):
			self._dq = np.zeros((nx, ny, nz), dtype=dtype)

		# Instantiate the stencil which does not clip the negative values
		self._stencil_diagnosing_density = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_density_defs,
			inputs={'in_d': self._d, 'in_q': self._q},
			outputs={'out_dq': self._dq},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

		# Instantiate the stencil which does clip the negative values
		self._stencil_diagnosing_and_clipping_density = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_and_clipping_density_defs,
			inputs={'in_d': self._d, 'in_q': self._q},
			outputs={'out_dq': self._dq},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	@staticmethod
	def _stencil_diagnosing_density_defs(in_d, in_q):
		"""
		GT4Py stencil diagnosing the density of the water constituent.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_q : gridtools.Equation
			The mass fraction of the water constituent.

		Return
		-------
		gridtools.Equation :
			The diagnosed density of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_dq = gt.Equation()

		# Computations
		out_dq[i, j, k] = in_d[i, j, k] * in_q[i, j, k]

		return out_dq

	@staticmethod
	def _stencil_diagnosing_and_clipping_density_defs(in_d, in_q):
		"""
		GT4Py stencil diagnosing the density of the water constituent,
		and then clipping the negative values.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_q : gridtools.Equation
			The mass fraction of the water constituent.

		Return
		-------
		gridtools.Equation :
			The diagnosed density of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output field
		tmp_dq = gt.Equation()
		out_dq = gt.Equation()

		# Computations
		tmp_dq[i, j, k] = in_d[i, j, k] * in_q[i, j, k]
		out_dq[i, j, k] = (tmp_dq[i, j, k] > 0.) * tmp_dq[i, j, k]

		return out_dq

	def _stencil_diagnosing_mass_fraction_initialize(self):
		"""
		Initialize the GT4Py stencils in charge of diagnosing the mass fraction
		of the water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_dq'):
			self._dq = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_q'):
			self._q = np.zeros((nx, ny, nz), dtype=dtype)

		# Instantiate the stencil which does not clip the negative values
		self._stencil_diagnosing_mass_fraction = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_mass_fraction_defs,
			inputs={'in_d': self._d, 'in_dq': self._dq},
			outputs={'out_q': self._q},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

		# Instantiate the stencil which does clip the negative values
		self._stencil_diagnosing_and_clipping_mass_fraction = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_and_clipping_mass_fraction_defs,
			inputs={'in_d': self._d, 'in_dq': self._dq},
			outputs={'out_q': self._q},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	@staticmethod
	def _stencil_diagnosing_mass_fraction_defs(in_d, in_dq):
		"""
		GT4Py stencil diagnosing the mass fraction of the water constituent.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_dq : gridtools.Equation
			The density of the water constituent.

		Return
		-------
		gridtools.Equation :
			The diagnosed mass fraction of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_q = gt.Equation()

		# Computations
		out_q[i, j, k] = in_dq[i, j, k] / in_d[i, j, k]

		return out_q

	@staticmethod
	def _stencil_diagnosing_and_clipping_mass_fraction_defs(in_d, in_dq):
		"""
		GT4Py stencil diagnosing the mass fraction of the water constituent,
		and then clipping the negative values.

		Parameters
		----------
		in_d : gridtools.Equation
			The air density.
		in_dq : gridtools.Equation
			The density of the water constituent.

		Return
		-------
		gridtools.Equation :
			The diagnosed mass fraction of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output field
		tmp_q = gt.Equation()
		out_q = gt.Equation()

		# Computations
		tmp_q[i, j, k] = in_dq[i, j, k] / in_d[i, j, k]
		out_q[i, j, k] = (tmp_q[i, j, k] > 0.) * tmp_q[i, j, k]

		return out_q
