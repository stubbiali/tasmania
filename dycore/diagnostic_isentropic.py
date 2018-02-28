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
import numpy as np

import gridtools as gt
from namelist import cp, datatype, g, p_ref, Rd

class DiagnosticIsentropic:
	"""
	Class implementing the diagnostic steps of the three-dimensional moist isentropic dynamical core
	using GT4Py's stencils.
	"""
	def __init__(self, grid, imoist, backend):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		imoist : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.
		"""
		self._grid, self._imoist, self._backend = grid, imoist, backend

		# The pointers to the stencil's compute function.
		# They will be initialized the first time the entry-point methods are invoked.
		self._stencil_diagnosing_momentums = None
		self._stencil_diagnosing_velocity_x = None
		self._stencil_diagnosing_velocity_y = None
		if self._imoist:
			self._stencil_diagnosing_water_constituents_mass = None
			self._stencil_diagnosing_water_constituents_mass_fraction = None
		self._stencil_diagnosing_pressure = None
		self._stencil_diagnosing_montgomery = None
		self._stencil_diagnosing_height = None

		# Allocate the Numpy array which will store the Exner function
		# Conversely to all other Numpy arrays carrying the output fields, this array is allocated
		# here as the Exner function, being a nonlinear function of the pressure,  can not be diagnosed 
		# via a GT4Py's stencil
		self._out_exn = np.zeros((grid.nx, grid.ny, grid.nz + 1), dtype = datatype)

		# Assign the corresponding z-level to each z-staggered grid point
		# This is required to diagnose the geometrical height at the half levels
		theta_1d = np.reshape(grid.z_half_levels.values[:, np.newaxis, np.newaxis], (1, 1, grid.nz + 1))
		self._theta = np.tile(theta_1d, (grid.nx, grid.ny, 1))

	def get_momentums(self, s, u, v):
		"""
		Diagnosis of the momentums :math:`U` and :math:`V`.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		u : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
		v : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the :math:`y`-velocity.

		Returns
		-------
		U : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`U`.
		V : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`V`.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_conservative_variables is None:
			self._initialize_stencil_diagnosing_momentums()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_momentums(s, u, v)

		# Run the stencil's compute function
		self._stencil_diagnosing_momentums.compute()

		return self._out_U, self._out_V

	def get_water_constituents_mass(self, s, qv, qc, qr):
		"""
		Diagnosis of the mass of each water constituent, i.e., :math:`Q_v`, :math:`Q_c` and :math:`Q_v`.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		qv : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of 
			water vapour.
		qc : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of 
			cloud water.
		qr : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of 
			precipitation water.

		Returns
		-------
		Qv : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_v`.
		Qc : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_c`.
		Qr : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`Q_r`.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_water_constituents_mass is None:
			self._initialize_stencil_diagnosing_water_constituents_mass()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_water_constituents_mass(s, qv, qc, qr)

		# Run the stencil's compute function
		self._stencil_diagnosing_water_constituents_mass.compute()

		return self._out_Qv, self._out_Qc, self._out_Qr

	def get_velocity_components(self, s, U, V):
		"""
		Diagnosis of the velocity components :math:`u` and :math:`v`.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		U : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
		V : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`y`-velocity.

		Returns
		-------
		u : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`u`.
		v : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the diagnosed :math:`v`.

		Note
		----
		The first and last rows (respectively, columns) of :data:`u` (resp., :data:`v`) are not set by the method.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_velocity_x is None:
			self._initialize_stencil_diagnosing_velocity_x()
			self._initialize_stencil_diagnosing_velocity_y()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencils_diagnosing_velocity(s, U, V)

		# Run the stencils' compute functions
		self._stencil_diagnosing_velocity_x.compute()
		self._stencil_diagnosing_velocity_y.compute()

		return self._out_u, self._out_v

	def get_water_constituents_mass_fraction(self, s, Qv, Qc, Qr):
		"""
		Diagnosis of the mass of each water constituents, i.e., :math:`q_v`, :math:`q_c` and :math:`q_r`.

		Parameters
		----------
		Qv : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of water vapour.
		Qc : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of cloud water.
		Qr : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass of precipitation water.

		Returns
		-------
		qv : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_v`.
		qc : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_c`.
		qr : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed :math:`q_r`.
		"""	
		# The first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil_diagnosing_water_constituents_mass_fraction is None:
			self._initialize_stencil_diagnosing_water_constituents_mass_fraction()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_water_constituents_mass_fraction(s, Qv, Qc, Qr)

		# Run the stencils' compute functions
		self._stencil_diagnosing_water_constituents_mass_fraction.compute()

		return self._out_Qv, self._out_Qc, self._out_Qr

	def get_diagnostic_variables(self, s, pt):
		"""
		Diagnosis of the pressure, the Exner function, the Montgomery potential, and the geometric height of the 
		potential temperature surfaces.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		pt : float 
			Boundary value for the pressure at the top of the domain.

		Returns
		-------
		p : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			pressure.
		exn : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			Exner function.
		mtg : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the diagnosed 
			Montgomery potential.
		h : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz+1`) representing the diagnosed 
			geometric height of the potential temperature surfaces.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_diagnosing_pressure is None:
			self._initialize_stencil_diagnosing_pressure()
			self._initialize_stencil_diagnosing_montgomery()
			self._initialize_stencil_diagnosing_height()

		# Update the attributes which serve as inputs to the GT4Py's stencils
		self._set_inputs_to_stencil_diagnosing_pressure(s)

		# Apply upper boundary condition for pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_pressure.compute()
	
		# Compute the Exner function
		# Note: the Exner function can not be computed via a GT4Py's stencils as it is a
		# nonlinear function of the pressure distribution
		self._out_exn[:, :, :] = cp * (self._out_p[:, :, :] / p_ref) ** (Rd / cp) 

		# Compute Montgomery potential at the lower main level
		mtg_s = self._grid.z_half_levels.values[-1] * self._out_exn[:, :, -1] + g * self._grid.topography_height
		self._out_mtg[:, :, -1] = mtg_s + 0.5 * self._grid.dz * self._out_exn[:, :, -1]

		# Compute Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery.compute()

		# Compute geometrical height of isentropes
		self._out_h[:, :, -1] = self._grid.topography_height
		self._stencil_diagnosing_height.compute()

		return self._out_p, self._out_exn, self._out_mtg, self._out_h


	def _initialize_stencil_diagnosing_momentums(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the momentums.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_u = np.zeros((nx + 1, ny, nz), dtype = datatype)
		self._in_v = np.zeros((nx, ny + 1, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_U = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_V = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_momentums = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_conservative_variables,
			inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v},
			outputs = {'out_U': self._out_U, 'out_V': self._out_V},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_momentums(self, s, u, v):	
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses the momentums.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		u : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx+1`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
		v : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny+1`, :obj:`nz`) representing the :math:`y`-velocity.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_u[:,:,:] = u[:,:,:]
		self._in_v[:,:,:] = v[:,:,:]

	def _defs_stencil_diagnosing_momentums(self, in_s, in_u, in_v): 
		"""
		GT4Py's stencil diagnosing the momentums :math:`U` and :math:`V`.

		Parameters
		----------
		in_s : obj 
			:class:`gridtools.Equation` representing the isentropic density.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		in_v : obj 
			:class:`gridtools.Equation` representing the :math:`y`-velocity.

		Returns
		-------
		out_U : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`U`.
		out_V : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`V`.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_U = gt.Equation()
		out_V = gt.Equation()

		# Computations
		out_U[i, j, k] = 0.5 * in_s[i, j, k] * (in_u[i, j, k] + in_u[i+1, j, k])
		out_V[i, j, k] = 0.5 * in_s[i, j, k] * (in_v[i, j, k] + in_v[i, j+1, k])

		return out_U, out_V


	def _initialize_stencil_diagnosing_water_constituents_mass(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the mass of each water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_Qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_Qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_Qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_water_constituents_mass = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_water_constituents_mass,
			inputs = {'in_s': self._in_s, 'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr},
			outputs = {'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_water_constituents_mass(self, s, qv, qc, qr):	
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses 
		the mass of each water constituent.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		qv : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of 
			water vapour.
		qc : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of
			cloud water.
		qr : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the mass fraction of
			precipitation water.
		"""
		self._in_s[:,:,:]  = s[:,:,:]
		self._in_qv[:,:,:] = qv[:,:,:]
		self._in_qc[:,:,:] = qc[:,:,:]
		self._in_qr[:,:,:] = qr[:,:,:]

	def _defs_stencil_diagnosing_water_constituents_mass(self, in_s, in_qv, in_qc, in_qr):
		"""
		GT4Py's stencil diagnosing the mass of each water constituent, i.e., :math:`Q_v`, :math:`Q_c` and :math:`Q_r`.

		Parameters
		----------
		in_s : obj 
			:class:`gridtools.Equation` representing the isentropic density.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapour.
		in_qc : obj 
			:class:`gridtools.Equation` representing the mass fraction of cloud water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		out_Qv : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qv`.
		out_Qc : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qc`.
		out_Qr : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`Qr`.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_Qv = gt.Equation()
		out_Qc = gt.Equation()
		out_Qr = gt.Equation()

		# Computations
		out_Qv[i, j, k] = in_s[i, j, k] * in_qv[i, j, k]
		out_Qc[i, j, k] = in_s[i, j, k] * in_qc[i, j, k]
		out_Qr[i, j, k] = in_s[i, j, k] * in_qr[i, j, k]

		return out_Qv, out_Qc, out_Qr


	def _initialize_stencil_diagnosing_velocity_x(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the :math:`x`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_U = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_u = np.zeros((nx + 1, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_x = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_x,
			inputs = {'in_s': self._in_s, 'in_U': self._in_U},
			outputs = {'out_u': self._out_u},
			domain = gt.domain.Rectangle((1, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _initialize_stencil_diagnosing_velocity_y(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the :math:`y`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_V = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_v = np.zeros((nx, ny + 1, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_y = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_velocity_y,
			inputs = {'in_s': self._in_s, 'in_V': self._in_V},
			outputs = {'out_v': self._out_v},
			domain = gt.domain.Rectangle((0, 1, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencils_diagnosing_velocity(self, s, U, V):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencils which diagnose the velocity components.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		U : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`x`-velocity.
		V : array_like 
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the :math:`y`-velocity.
		"""
		self._in_s[:,:,:] = s[:,:,:]
		self._in_U[:,:,:] = U[:,:,:]
		self._in_V[:,:,:] = V[:,:,:]

	def _defs_stencil_diagnosing_velocity_x(self, in_s, in_U):
		"""
		GT4Py's stencil diagnosing the :math:`x`-component of the velocity.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`x`-velocity.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_u = gt.Equation()

		# Computations
		out_u[i, j, k] = (in_U[i-1, j, k] + in_U[i, j, k]) / (in_s[i-1, j, k] + in_s[i, j, k])

		return out_u

	def _defs_stencil_diagnosing_velocity_y(self, in_s, in_V):
		"""
		GT4Py's stencil diagnosing the :math:`y`-component of the velocity.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`y`-velocity.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_v = gt.Equation()

		# Computations
		out_v[i, j, k] = (in_V[i, j-1, k] + in_V[i, j, k]) / (in_s[i, j-1, k] + in_s[i, j, k])
			
		return out_v


	def _initialize_stencil_diagnosing_water_constituents_mass_fraction(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the mass fraction of each water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will carry the input fields
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_Qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will carry the output fields
		self._out_qv = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qc = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qr = np.zeros((nx, ny, nz), dtype = datatype)

		# Instantiate the stencil
		self._stencil_diagnosing_water_constituents_mass_fraction = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_water_constituents_mass_fraction,
			inputs = {'in_s': self._in_s, 'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr},
			outputs = {'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)), 
			mode = self._backend)

	def _set_inputs_to_stencil_diagnosing_water_constituents_mass_fraction(self, s, Qv, Qc, Qr):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnose 
		the mass fraction of each water constituent.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		Qc : obj 
			:class:`gridtools.Equation` representing the mass of cloud water.
		Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.
		"""
		self._in_s[:,:,:]  = s[:,:,:]
		self._in_Qv[:,:,:] = Qv[:,:,:]
		self._in_Qc[:,:,:] = Qc[:,:,:]
		self._in_Qr[:,:,:] = Qr[:,:,:]


	def _defs_stencil_diagnosing_water_constituents_mass_fraction(self, in_s, in_Qv, in_Qc, in_Qr):
		"""
		GT4Py's stencil diagnosing the mass fraction of each water constituent.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		in_Qv : obj
			:class:`gridtools.Equation` representing the mass of water vapour.
		in_Qc : obj
			:class:`gridtools.Equation` representing the mass of cloud water.
		in_Qr : obj
			:class:`gridtools.Equation` representing the mass of precipitation water.

		Returns
		-------
		out_qv : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of water vapour.
		out_qc : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of cloud water.
		out_qr : obj
			:class:`gridtools.Equation` representing the diagnosed mass fraction of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()
		out_qr = gt.Equation()

		# Computations
		out_qv[i, j, k] = in_Qv[i, j, k] / in_s[i, j, k]
		out_qc[i, j, k] = in_Qc[i, j, k] / in_s[i, j, k]
		out_qr[i, j, k] = in_Qr[i, j, k] / in_s[i, j, k]

		return out_qv, out_qc, out_qr


	def _initialize_stencil_diagnosing_pressure(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the pressure.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the input field
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will carry the output field
		self._out_p = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._in_p = self._out_p

		# Instantiate the stencil
		self._stencil_diagnosing_pressure = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_pressure,
			inputs = {'in_s': self._in_s, 'in_p': self._in_p},
			outputs = {'out_p': self._out_p},
			domain = gt.domain.Rectangle((0, 0, 1), (nx - 1, ny - 1, nz)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.FORWARD)

	def _initialize_stencil_diagnosing_montgomery(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the Montgomery potential.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the output field
		self._out_mtg = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_mtg = self._out_mtg

		# Instantiate the stencil
		self._stencil_diagnosing_montgomery = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_montgomery,
			inputs = {'in_exn': self._out_exn, 'in_mtg': self._in_mtg},
			outputs = {'out_mtg': self._out_mtg},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 2)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.BACKWARD)
	
	def _initialize_stencil_diagnosing_height(self):
		"""
		Initialize the GT4Py's stencil in charge of diagnosing the geometric height of the half-level isentropes.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will carry the output field
		self._out_h = np.zeros((nx, ny, nz + 1), dtype = datatype)
		self._in_h = self._out_h

		# Instantiate the stencil
		self._stencil_diagnosing_height = gt.NGStencil( 
			definitions_func = self._defs_stencil_diagnosing_height,
			inputs = {'in_theta': self._theta, 'in_exn': self._out_exn, 'in_p': self._out_p, 'in_h': self._in_h},
			outputs = {'out_h': self._out_h},
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend,
			vertical_direction = gt.vertical_direction.BACKWARD)
	
	def _set_inputs_to_stencil_diagnosing_pressure(self, s):
		"""
		Update the private instance attributes which serve as inputs to the GT4Py's stencil which diagnoses the pressure.

		Parameters
		----------
		s : array_like
			:class:`numpy.ndarray` with shape (:obj:`nx`, :obj:`ny`, :obj:`nz`) representing the isentropic density.
		"""
		self._in_s[:,:,:] = s[:,:,:]
	
	def _defs_stencil_diagnosing_pressure(self, in_s, in_p):
		"""
		GT4Py's stencil diagnosing the pressure.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed pressure.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_p = gt.Equation()

		# Computations
		out_p[i, j, k] = in_p[i, j, k-1] + g * self._grid.dz * in_s[i, j, k-1]

		return out_p

	def _defs_stencil_diagnosing_montgomery(self, in_exn, in_mtg):
		"""
		GT4Py's stencil diagnosing the Exner function.

		Parameters
		----------
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed Montgomery potential.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_mtg = gt.Equation()

		# Computations
		out_mtg[i, j, k] = in_mtg[i, j, k+1] + self._grid.dz * in_exn[i, j, k+1]

		return out_mtg

	def _defs_stencil_diagnosing_height(self, in_theta, in_exn, in_p, in_h):
		"""
		GT4Py's stencil diagnosing the geometric height of the isentropes.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the vertical half levels.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the isentropes.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed geometric height of the isentropes.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_h = gt.Equation()

		# Computations
		out_h[i, j, k] = in_h[i, j, k+1] - Rd * (in_theta[i, j, k  ] * in_exn[i, j, k  ] +
												 in_theta[i, j, k+1] * in_exn[i, j, k+1]) * \
												(in_p[i, j, k] - in_p[i, j, k+1]) / \
												(cp * g * (in_p[i, j, k] + in_p[i, j, k+1]))

		return out_h

