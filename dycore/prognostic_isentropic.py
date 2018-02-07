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
Classes implementing different schemes to carry out the prognostic step of the three-dimensional 
moist isentropic dynamical core.
"""
import abc
import numpy as np

from dycore.flux_isentropic import FluxIsentropic
import gridtools as gt
from namelist import datatype

class PrognosticIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes to carry out the prognostic step of 
	the three-dimensional moist isentropic dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, imoist, scheme, backend):
		"""
		Constructor.

		Parameters
		----------
			grid : obj 
				:class:`~grids.xyz_grid.XYZGrid` representing the underlying grid.
			imoist : bool 
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			scheme : str 
				String specifying the scheme to use. Either:
					* 'upwind', for the upwind scheme;
					* 'leapfrog', for the leapfrog scheme;
					* 'maccormack', for the MacCormack scheme.
			backend : obj 
				:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.
		"""
		self._grid, self._imoist, self._scheme, self._backend = grid, imoist, scheme, backend
		self._flux = FluxIsentropic.factory(scheme, grid, imoist)

	@abc.abstractmethod
	def step_forward(self, diagnostic, boundary, dt, s, u, v, p, mtg, U, V, Qv = None, Qc = None, Qr = None,
			   	     old_s = None, old_U = None, old_V = None, old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		Method advancing the conservative model variables one time step forward.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
			diagnostic : obj 
				:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` performing the diagnostic steps
				of the dynamical core.
			boundary : obj 
				An instance of one of the derived classes of :class:`~dycore.horizontal_boundary.HorizontalBoundary`,
				implementing the lateral boundary conditions.
			dt : obj 
				A :class:`datetime.timedelta` representing the time step.
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
			p : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
			mtg : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
				at current time.
			U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
				at current time.
			V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at current time.
			Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at current time.
			Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at current time.
			Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at current time.
			old_s : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the previous time level.
			old_U : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the previous time level.
			old_V : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at the previous time level.
			old_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at the previous time level.
			old_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at the previous time level.
			old_Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
				at the previous time level.

		Returns
		-------
			out_s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the next time level.
			out_U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the next time level.
			out_V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum
				at the next time level.
			out_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour
				at the next time level.
			out_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water
				at the next time level.
			out_Qr : `array_like`, optional
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at the next time level.
		"""

	@property
	def nb(self):
		"""
		Get the number of boundary layers.

		Return
		------
			int :
			 	The number of boundary layers.
		"""
		return self._flux.nb

	@staticmethod
	def factory(grid, imoist, scheme, backend):
		"""
		Static method returning an instace of the derived class implementing the scheme specified by :data:`scheme`.

		Parameters
		----------
			grid : obj 
				:class:`~grids.xyz_grid.XYZGrid` representing the underlying grid.
			imoist : bool 
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			scheme : str 
				String specifying the scheme to use. Either:
					* 'upwind', for the upwind scheme;
					* 'leapfrog', for the leapfrog scheme;
					* 'maccormack', for the MacCormack scheme.
			backend : obj 
				:class:`gridtools.Mode` specifying the backend for the GT4Py's stencils.

		Return
		------
			obj :
				An instace of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if scheme in ['upwind', 'maccormack']:
			return IsentropicPrognosticTL1(grid, imoist, scheme, backend)
		elif scheme in ['leapfrog']:
			return IsentropicPrognosticTL2(grid, imoist, scheme, backend)
		else:
			raise ValueError('Unknown scheme.')

	def _allocate_inputs(self, s, u, v):
		"""
		Allocate (some of) the private instance attributes which will serve as inputs to the GT4Py's stencils.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Instantiate a GT4Py's Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will carry the input fields
		self._in_s   = np.zeros_like(s)
		self._in_u   = np.zeros_like(u)
		self._in_v   = np.zeros_like(v)
		self._in_mtg = np.zeros_like(s)
		self._in_U   = np.zeros_like(s)
		self._in_V   = np.zeros_like(s)
		if self._imoist:
			self._in_Qv = np.zeros_like(s)
			self._in_Qc = np.zeros_like(s)
			self._in_Qr = np.zeros_like(s)

	def _allocate_outputs(self, s):
		"""
		Allocate the Numpy arrays which will store the updated solution.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density.
		"""
		# Determine the extent of the computational domain based on the input Numpy arrays. 
		# Note that, compared to the DataArrays carrying the state fields, these arrays may be larger, 
		# as they might have been decorated with some extra layers to accommodate the 
		# horizontal boundary conditions.
		self.ni = s.shape[0] - 2 * self.nb
		self.nj = s.shape[1] - 2 * self.nb
		self.nk = s.shape[2]

		# Allocate the Numpy arrays which will carry the output fields
		# Note: allocation is performed here, i.e., the first time the entry-point method is invoked,
		# so to make this step independent of the boundary conditions type
		self._out_s = np.zeros_like(s)
		self._out_U = np.zeros_like(s)
		self._out_V = np.zeros_like(s)
		if self._imoist:
			self._out_Qv = np.zeros_like(s)
			self._out_Qc = np.zeros_like(s)
			self._out_Qr = np.zeros_like(s)

	def _set_inputs(self, dt, s, u, v, mtg, U, V, Qv, Qc, Qr):
		"""
		Update (some of) the attributes which serve as inputs to the GT4Py's stencils.

		Parameters
		----------
			dt : obj 
				A :class:`datetime.timedelta` representing the time step.
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
			p : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
			mtg : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
				at current time.
			U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
				at current time.
			V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at current time.
			Qv : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at current time.
			Qc : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at current time.
			Qr : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at current time.
		"""
		# Time step
		self._dt.value = dt.seconds
		
		# Current state
		self._in_s[:,:,:]   = s[:,:,:]
		self._in_u[:,:,:]   = u[:,:,:]
		self._in_v[:,:,:]   = v[:,:,:]
		self._in_mtg[:,:,:] = mtg[:,:,:]
		self._in_U[:,:,:]   = U[:,:,:]
		self._in_V[:,:,:]   = V[:,:,:]
		if self._imoist:
			self._in_Qv[:,:,:] = Qv[:,:,:]
			self._in_Qc[:,:,:] = Qc[:,:,:]
			self._in_Qr[:,:,:] = Qr[:,:,:]
		

class PrognosticIsentropicTL1(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement 
	a standard one-time-level scheme carrying out the prognostic step of the three-dimensional 
	moist isentropic dynamical core.

	Attributes
	----------
		time_levels : int
			Number of time levels the scheme relies on.
		steps : int
			Number of steps the scheme entails.
	"""
	def __init__(self, grid, imoist, scheme, backend):
		"""
		Constructor.
		
		Parameters
		----------
			grid : obj
				:class:`~grids.xyz_grid.XYZGrid` representing the underlying grid.
			imoist : bool 
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			scheme : str 
				String specifying the one-time-level scheme to use. Either:
					* 'upwind', for the upwind scheme;
					* 'maccormack', for the MacCormack scheme.
			backend : obj
				:class:`gridtools.mode` specifying the backend for the GT$Py's stencils.

		Note
		----
			To instantiate an object of this class, one should prefer the static method
			:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of
			:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(grid, imoist, scheme, backend)

		# As the number of steps, also the number of boundary layers, hence the extent of the
		# computational domain, is scheme-dependent
		self.time_levels = 1
		if self._scheme in ['upwind', 'maccormack']:
			self.steps = 1

		# The pointers to the stencils' compute function
		# They will be re-directed the first time the forward method is invoked
		self._stencil_isentropic_density_and_water_constituents = None
		self._stencil_momentums = None

	def step_forward(self, diagnostic, boundary, dt, s, u, v, p, mtg, U, V, Qv = None, Qc = None, Qr = None,
			   	     old_s = None, old_U = None, old_V = None, old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		Method advancing the conservative model variables one time step forward via a one-time-level scheme.

		Parameters
		----------
			diagnostic : obj 
				:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` performing the diagnostic steps
				of the dynamical core.
			boundary : obj 
				An instance of one of the derived classes of :class:`~dycore.horizontal_boundary.HorizontalBoundary`,
				implementing the lateral boundary conditions.
			dt : obj 
				A :class:`datetime.timedelta` representing the time step.
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
			p : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
			mtg : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
				at current time.
			U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
				at current time.
			V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at current time.
			Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at current time.
			Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at current time.
			Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at current time.
			old_s : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the previous time level.
			old_U : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the previous time level.
			old_V : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at the previous time level.
			old_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at the previous time level.
			old_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at the previous time level.
			old_Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
				at the previous time level.

		Returns
		-------
			out_s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the next time level.
			out_U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the next time level.
			out_V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum
				at the next time level.
			out_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour
				at the next time level.
			out_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water
				at the next time level.
			out_Qr : `array_like`, optional
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at the next time level.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_isentropic_density_and_water_constituents is None:
			self._initialize_stencils(s, u, v)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs(dt, s, u, v, mtg, U, V, Qv, Qc, Qr)
		
		# Run the compute function of the stencil stepping the isentropic density and the water constituents,
		# and providing provisional values for the momentums
		self._stencil_isentropic_density_and_water_constituents.compute()

		# Apply the boundary conditions on the current and updated isentropic density
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		now_s = boundary.from_computational_to_physical_domain(s, (nx, ny, nz))
		new_s = boundary.from_computational_to_physical_domain(self._out_s, (nx, ny, nz))
		boundary.apply(new_s, now_s)

		# Compute the predicted isentropic density after half time step, then get the associated stencils'
		# computational domain
		half_s = 0.5 * (now_s + new_s) 
		self._half_s[:,:,:] = boundary.from_physical_to_computational_domain(half_s)

		# Diagnose the Montgomery potential after half time step, then get the associated stencils'
		# computational domain
		_, _, half_mtg, _ = diagnostic.get_diagnostic_variables(half_s, p[0,0,0])
		self._half_mtg[:,:,:] = boundary.from_physical_to_computational_domain(half_mtg)

		# Run the compute function of the stencil stepping the momentums
		self._stencil_momentums.compute()

		if not self._imoist:
			return self._out_s, self._out_U, self._out_V
		return self._out_s, self._out_U, self._out_V, self._out_Qv, self._out_Qc, self._out_Qr

	def _initialize_stencils(self, s, u, v):
		"""
		Initialize the GT4Py's stencil implementing the one-time-level scheme.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
		"""
		# Allocate the Numpy arrays which will serve as inputs to the first stencil
		self._allocate_inputs(s, u, v)

		# Allocate the Numpy arrays which will store temporary fields
		self._allocate_temporaries(s)

		# Allocate the Numpy arrays which will carry the output fields
		self._allocate_outputs(s)

		# Set the computational domain and the backend
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + self.ni - 1, self.nb + self.nj - 1, self.nk - 1))
		_mode = self._backend

		# Instantiate the first stencil
		if not self._imoist:
			self._stencil_isentropic_density_and_water_constituents = gt.NGStencil( 
				definitions_func = self._defs_stencil_isentropic_density_and_water_constituents,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._tmp_U, 'out_V': self._tmp_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_isentropic_density_and_water_constituents = gt.NGStencil( 
				definitions_func = self._defs_stencil,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,  
						  'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._tmp_U, 'out_V': self._tmp_V,
						   'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
				domain = _domain, 
				mode = _mode)

		# Instantiate the second stencil
		self._stencil_momentums = gt.NGStencil( 
			definitions_func = self._defs_stencil_momentums,
			inputs = {'in_s': self._half_s, 'in_mtg': self._half_mtg, 'in_U': self._tmp_U, 'in_V': self._tmp_V},
			global_inputs = {'dt': self._dt},
			outputs = {'out_U': self._out_U, 'out_V': self._out_V},
			domain = _domain, 
			mode = _mode)

	def _allocate_temporaries(self, s):
		"""
		Allocate the Numpy arrays which will store temporary fields.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
		"""
		self._tmp_U    = np.zeros_like(s)
		self._tmp_V    = np.zeros_like(s)
		self._half_s   = np.zeros_like(s)
		self._half_mtg = np.zeros_like(s)

	def _defs_stencil_isentropic_density_and_water_constituents(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  											in_Qv = None, in_Qc = None, in_Qr = None):
		"""
		GT4Py's stencil stepping the isentropic density and the water constituents via a one-time-level scheme.
		Further, it computes the provisional values for the momentums.
		
		Parameters
		----------
			dt : obj
				:class:`gridtools.Global` representing the time step.
			in_s : obj
				:class:`gridtools.Equation` representing the isentropic density at the current time.
			in_u : obj
				:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time.
			in_v : obj
				:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time.
			in_mtg : obj
				:class:`gridtools.Equation` representing the Montgomery potential at the current time.
			in_U : obj
				:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
			in_V : obj
				:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
			in_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the mass of water vapour at the current time.
			in_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the mass of cloud water at the current time.
			in_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the mass of precipitation water at the current time.

		Returns
		-------
			out_s : obj
				:class:`gridtools.Equation` representing the stepped isentropic density.
			out_U : obj
				:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
			out_V : obj
				:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
			out_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of water vapour.
			out_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of cloud water.
			out_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._imoist:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		if not self._imoist:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y = \
				self._flux.get_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y,
			flux_Qv_x, flux_Qv_y, flux_Qc_x, flux_Qc_y, flux_Qr_x, flux_Qr_y = \
				self._flux.get_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr)

		out_s[i, j, k] = in_s[i, j, k] - dt * ((flux_s_x[i+1, j, k] - flux_s_x[i, j, k]) / self._grid.dx +
						 					   (flux_s_y[i, j+1, k] - flux_s_y[i, j, k]) / self._grid.dy)
		out_U[i, j, k] = in_U[i, j, k] - dt * ((flux_U_x[i+1, j, k] - flux_U_x[i, j, k]) / self._grid.dx +
						 					   (flux_U_y[i, j+1, k] - flux_U_y[i, j, k]) / self._grid.dy)
		out_V[i, j, k] = in_V[i, j, k] - dt * ((flux_V_x[i+1, j, k] - flux_V_x[i, j, k]) / self._grid.dx +
						 					   (flux_V_y[i, j+1, k] - flux_V_y[i, j, k]) / self._grid.dy)
		if self._imoist:
			out_Qv[i, j, k] = in_Qv[i, j, k] - dt * ((flux_Qv_x[i+1, j, k] - flux_Qv_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qv_y[i, j+1, k] - flux_Qv_y[i, j, k]) / self._grid.dy)
			out_Qc[i, j, k] = in_Qc[i, j, k] - dt * ((flux_Qc_x[i+1, j, k] - flux_Qc_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qc_y[i, j+1, k] - flux_Qc_y[i, j, k]) / self._grid.dy)
			out_Qr[i, j, k] = in_Qr[i, j, k] - dt * ((flux_Qr_x[i+1, j, k] - flux_Qr_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qr_y[i, j+1, k] - flux_Qr_y[i, j, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def _defs_stencil_momentums(self, dt, in_s, in_mtg, in_U, in_V):
		"""
		GT4Py's stencil stepping the momentums via a one-time-level scheme.

		Parameters
		----------
			dt : obj
				:class:`gridtools.Global` representing the time step.
			in_s : obj
				:class:`gridtools.Equation` representing the stepped isentropic density.
			in_mtg : obj
				:class:`gridtools.Equation` representing the Montgomery potential diagnosed from the stepped isentropic density.
			in_U : obj
				:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
			in_V : obj
				:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.

		Returns
		-------
			out_U : obj
				:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
			out_V : obj
				:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_U = gt.Equation()
		out_V = gt.Equation()

		# Computations
		out_U[i, j, k] = in_U[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = in_V[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy

		return out_U, out_V


class PrognosticIsentropicTL2(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to 
	implement a standard two-time-levels scheme carrying out the prognostic step of the three-dimensional 
	moist isentropic dynamical core. An example of this kind of schemes is the leapfrog scheme.

	Attributes
	----------
		time_levels : int
			Number of time levels the scheme relies on.
		steps : int
			Number of steps the scheme entails.
	"""
	def __init__(self, grid, imoist, scheme, backend):
		"""
		Constructor.
		
		Parameters
		----------
			grid : obj
				:class:`~grids.xyz_grid.XYZGrid` representing the underlying grid.
			imoist : bool 
				:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			scheme : str 
				String specifying the two-time-level scheme to use. Either:
					* 'leapfrog', for the leapfrog scheme.
			backend : obj
				:class:`gridtools.mode` specifying the backend for the GT$Py's stencils.

		Note
		----
			To instantiate an object of this class, one should prefer the static method
			:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of
			:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(grid, imoist, scheme, backend)

		# As the number of steps, also the number of boundary layers, hence the extent of the
		# computational domain, is scheme-dependent
		self.time_levels = 2
		if self._scheme in ['leapfrog']:
			self.steps = 1

		# The pointer to the stencil's compute function
		# It will be re-directed the first time the forward method is invoked
		self._stencil = None

	def step_forward(self, boundary, diagnostic, dt, s, u, v, p, mtg, U, V, Qv = None, Qc = None, Qr = None,
			   	     old_s = None, old_U = None, old_V = None, old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		Method advancing the conservative model variables one time step forward via a two-time-level scheme.

		Parameters
		----------
			diagnostic : obj 
				:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` performing the diagnostic steps
				of the dynamical core.
			boundary : obj 
				An instance of one of the derived classes of :class:`~dycore.horizontal_boundary.HorizontalBoundary`,
				implementing the lateral boundary conditions.
			dt : obj 
				A :class:`datetime.timedelta` representing the time step.
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
			p : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
			mtg : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
				at current time.
			U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
				at current time.
			V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at current time.
			Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at current time.
			Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at current time.
			Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at current time.
			old_s : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the previous time level.
			old_U : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the previous time level.
			old_V : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at the previous time level.
			old_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at the previous time level.
			old_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at the previous time level.
			old_Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
				at the previous time level.

		Returns
		-------
			out_s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the next time level.
			out_U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the next time level.
			out_V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum
				at the next time level.
			out_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour
				at the next time level.
			out_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water
				at the next time level.
			out_Qr : `array_like`, optional
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at the next time level.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil is None:
			self._initialize_stencil(s, u, v)

		# Update the attributes which serve as inputs to the GT4Py's stencil
		self._set_inputs(dt, s, u, v, mtg, U, V, Qv, Qc, Qr, old_s, old_U, old_V, old_Qv, old_Qc, old_Qr)
		
		# Run the stencil's compute function
		self._stencil.compute()

		if not self._imoist:
			return self._out_s, self._out_U, self._out_V
		return self._out_s, self._out_U, self._out_V, self._out_Qv, self._out_Qc, self._out_Qr

	def _initialize_stencil(self, s, u, v):
		"""
		Initialize the GT4Py's stencil implementing the two-time-levels scheme.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._allocate_inputs(s, u, v)

		# Allocate the Numpy arrays which will carry the output fields
		self._allocate_outputs(s)

		# Set the computational domain and the backend
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + self.ni - 1, self.nb + self.nj - 1, self.nk - 1))
		_mode = self._backend

		# Instantiate the stencil
		if not self._imoist:
			self._stencil = gt.NGStencil( 
				definitions_func = self._defs_stencil,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,
						  'old_s': self._old_s, 'old_U': self._old_U, 'old_V': self._old_V},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._out_U, 'out_V': self._out_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil = gt.NGStencil( 
				definitions_func = self._defs_stencil,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,
						  'old_s': self._old_s, 'old_U': self._old_U, 'old_V': self._old_V,
						  'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr,
						  'old_Qv': self._old_Qv, 'old_Qc': self._old_Qc, 'old_Qr': self._old_Qr},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._out_U, 'out_V': self._out_V,
						   'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
				domain = _domain, 
				mode = _mode)

	def _allocate_inputs(self, s, u, v):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py's stencil.

		Parameters
		----------
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Instantiate a GT4Py's Global representing the timestep and the Numpy arrays
		# which will carry the solution at the current time step
		super()._allocate_inputs(s, u, v)

		# Allocate the Numpy arrays which will carry the solution at the previous time step
		self._old_s = np.zeros_like(s)
		self._old_U = np.zeros_like(s)
		self._old_V = np.zeros_like(s)
		if self._imoist:
			self._old_Qv = np.zeros_like(s)
			self._old_Qc = np.zeros_like(s)
			self._old_Qr = np.zeros_like(s)

	def _set_inputs(self, dt, s, u, v, mtg, U, V, Qv, Qc, Qr, old_s, old_U, old_V, old_Qv, old_Qc, old_Qr):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencil.

		Parameters
		----------
			dt : obj 
				A :class:`datetime.timedelta` representing the time step.
			s : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
				at current time.
			u : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
				at current time.
			v : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
				at current time.
			p : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
			mtg : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
				at current time.
			U : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
				at current time.
			V : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at current time.
			Qv : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at current time.
			Qc : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at current time.
			Qr : array_like 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
				at current time.
			old_s : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
				at the previous time level.
			old_U : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
				at the previous time level.
			old_V : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
				at the previous time level.
			old_Qv : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
				at the previous time level.
			old_Qc : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
				at the previous time level.
			old_Qr : `array_like`, optional 
				:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
				at the previous time level.
		"""
		# Update the time step and the Numpy arrays carrying the current solution
		super()._set_inputs(dt, s, u, v, mtg, U, V, Qv, Qc, Qr)
		
		# Update the Numpy arrays carrying the solution at the previous time step
		self._old_s[:,:,:] = old_s[:,:,:]
		self._old_U[:,:,:] = old_U[:,:,:]
		self._old_V[:,:,:] = old_V[:,:,:]
		if self._imoist:
			self._old_Qv[:,:,:] = old_Qv[:,:,:]
			self._old_Qc[:,:,:] = old_Qc[:,:,:]
			self._old_Qr[:,:,:] = old_Qr[:,:,:]

	def _defs_stencil(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  in_Qv = None, in_Qc = None, in_Qr = None,
					  old_s = None, old_U = None, old_V = None, 
					  old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		GT4Py's stencil implementing a two-time-levels scheme.

		Parameters
		----------
			dt : obj
				:class:`gridtools.Global` representing the time step.
			in_s : obj
				:class:`gridtools.Equation` representing the isentropic density at the current time.
			in_u : obj
				:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time.
			in_v : obj
				:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time.
			in_mtg : obj
				:class:`gridtools.Equation` representing the Montgomery potential at the current time.
			in_U : obj
				:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
			in_V : obj
				:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
			in_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the mass of water vapour at the current time.
			in_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the mass of cloud water at the current time.
			in_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the mass of precipitation water at the current time.
			old_s : `obj`, optional
				:class:`gridtools.Equation` representing the isentropic density at the previous time level.
			old_U : `obj`, optional
				:class:`gridtools.Equation` representing the :math:`x`-momentum at the previous time level.
			old_V : `obj`, optional
				:class:`gridtools.Equation` representing the :math:`y`-momentum at the previous time level.
			old_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the mass of water vapour at the previous time level.
			old_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the mass of cloud water at the previous time level.
			old_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the mass of precipitation water at the previous time level.

		Returns
		-------
			out_s : obj
				:class:`gridtools.Equation` representing the stepped isentropic density.
			out_U : obj
				:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
			out_V : obj
				:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
			out_Qv : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of water vapour.
			out_Qc : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of cloud water.
			out_Qr : `obj`, optional
				:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._imoist:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		if not self._imoist:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y = \
				self._flux.get_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y,
			flux_Qv_x, flux_Qv_y, flux_Qc_x, flux_Qc_y, flux_Qr_x, flux_Qr_y = \
				self._flux.get_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr)

		out_s[i, j, k] = old_s[i, j, k] - 2. * dt * ((flux_s_x[i+1, j, k] - flux_s_x[i, j, k]) / self._grid.dx + \
						 							 (flux_s_y[i, j+1, k] - flux_s_y[i, j, k]) / self._grid.dy)  
		out_U[i, j, k] = old_U[i, j, k] - 2. * dt * ((flux_U_x[i+1, j, k] - flux_U_x[i, j, k]) / self._grid.dx + \
						 							 (flux_U_y[i, j+1, k] - flux_U_y[i, j, k]) / self._grid.dy)  \
										- dt * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = old_V[i, j, k] - 2. * dt * ((flux_V_x[i+1, j, k] - flux_V_x[i, j, k]) / self._grid.dx + \
						 							 (flux_V_y[i, j+1, k] - flux_V_y[i, j, k]) / self._grid.dy)  \
										- dt * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy
		if self._imoist:
			out_Qv[i, j, k] = old_Qv[i, j, k] - 2. * dt * ((flux_Qv_x[i+1, j, k] - flux_Qv_x[i, j, k]) / self._grid.dx + \
						 						  	   	   (flux_Qv_y[i, j+1, k] - flux_Qv_y[i, j, k]) / self._grid.dy)
			out_Qc[i, j, k] = old_Qc[i, j, k] - 2. * dt * ((flux_Qc_x[i+1, j, k] - flux_Qc_x[i, j, k]) / self._grid.dx + \
						 						 	   	   (flux_Qc_y[i, j+1, k] - flux_Qc_y[i, j, k]) / self._grid.dy)
			out_Qr[i, j, k] = old_Qr[i, j, k] - 2. * dt * ((flux_Qr_x[i+1, j, k] - flux_Qr_x[i, j, k]) / self._grid.dx + \
						 						  	   	   (flux_Qr_y[i, j+1, k] - flux_Qr_y[i, j, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V


