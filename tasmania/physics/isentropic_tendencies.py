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
	NonconservativeIsentropicPressureGradient
	ConservativeIsentropicPressureGradient
	VerticalIsentropicAdvection
	PrescribedSurfaceHeating
"""
import numpy as np
from sympl import DataArray, TendencyComponent

import gridtools as gt
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.isentropic_fluxes import VerticalHomogeneousIsentropicFlux
from tasmania.utils.data_utils import get_physical_constants

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class NonconservativeIsentropicPressureGradient(TendencyComponent):
	"""
	This class calculates the anti-gradient of the Montgomery potential,
	which provides tendencies for the :math:`x`- and :math:`y`-velocity
	in the isentropic system.
	"""
	def __init__(self, grid, order, horizontal_boundary_type,
				 backend=gt.mode.NUMPY, dtype=datatype, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		order : int
			The order of the finite difference formula used to
			discretized the gradient of the Montgomery potential. Either:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		# Keep track of input parameters
		self._grid     = grid
		self._order	   = order
		self._backend  = backend
		self._dtype	   = dtype

		# Call parent's constructor
		super().__init__(**kwargs)

		# Instantiate the class taking care of the lateral boundary conditions
		self._hboundary = HorizontalBoundary.factory(horizontal_boundary_type,
													 grid, self.nb)

		# Initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery
		# potential; it will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'x_velocity':
				{'dims': dims, 'units': 'm s^-2'},
			'y_velocity':
				{'dims': dims, 'units': 'm s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		"""
		Returns
		-------
		int :
			Number of halo layers in the horizontal directions.
		"""
		if self._order == 2:
			return 1
		elif self._order == 4:
			return 2
		else:
			import warnings
			warnings.warn('Order {} not supported; set order to 2.'.format(self._order))
			self._order = 2
			return 1

	@property
	def _stencil_defs(self):
		if self._order == 2:
			return self._stencil_second_order_defs
		elif self._order == 4:
			return self._stencil_fourth_order_defs

	def array_call(self, state):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Instantiate the GT4Py stencil calculating the pressure gradient
		if self._stencil is None:
			self._stencil_initialize()

		# Extend/shrink the Montgomery potential to accommodate for the
		# lateral boundary conditions
		mtg = state['montgomery_potential']
		self._in_mtg[...] = self._hboundary.from_physical_to_computational_domain(mtg)

		# Run the stencil
		self._stencil.compute()

		# Bring the stencil's outputs back to the physical domain shape.
		# Note that we do not enforce the boundary conditions on the
		# Montgomery potential. Therefore, in the case of relaxed boundary
		# conditions, the outermost halo layers in the output fields would
		# be zero. Nevertheless, periodic conditions are applied exactly.
		u_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_u_tnd, (nx, ny, nz))
		v_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_v_tnd, (nx, ny, nz))

		# Set the other return dictionary
		tendencies = {'x_velocity': u_tnd, 'y_velocity': v_tnd}

		return tendencies, {}

	def _stencil_initialize(self):
		# Shortcuts
		mi, mj, mk = self._hboundary.mi, self._hboundary.mj, self._grid.nz
		nb = self.nb

		# Allocate the NumPy arrays which serve as stencil's input
		self._in_mtg = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Allocate the NumPy arrays which serve as stencil's output
		self._out_u_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._out_v_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_mtg': self._in_mtg},
			outputs			 = {'out_u_tnd': self._out_u_tnd,
								   'out_v_tnd': self._out_v_tnd},
			domain			 = gt.domain.Rectangle((nb, nb, 0),
													 (mi-nb-1, mj-nb-1, mk-1)),
			mode			 = self._backend
		)

	def _stencil_second_order_defs(self, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# Define the computations
		out_u_tnd[i, j] = (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_v_tnd[i, j] = (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_u_tnd, out_v_tnd

	def _stencil_fourth_order_defs(self, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_u_tnd = gt.Equation()
		out_v_tnd = gt.Equation()

		# Define the computations
		out_u_tnd[i, j] = (- in_mtg[i-2, j] + 8. * in_mtg[i-1, j]
						   - 8. * in_mtg[i+1, j] + in_mtg[i+2, j]) / (12. * dx)
		out_v_tnd[i, j] = (- in_mtg[i, j-2] + 8. * in_mtg[i, j-1]
						   - 8. * in_mtg[i, j+1] + in_mtg[i, j+2]) / (12. * dy)

		return out_u_tnd, out_v_tnd


class ConservativeIsentropicPressureGradient(TendencyComponent):
	"""
	This class calculates the anti-gradient of the Montgomery potential,
	multiplied by the air isentropic density. This quantity provides
	tendencies for the :math:`x`- and :math:`y`-momentum in the
	isentropic system.
	"""
	def __init__(self, grid, order, horizontal_boundary_type,
				 backend=gt.mode.NUMPY, dtype=datatype, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		order : int
			The order of the finite difference formula used to
			discretized the gradient of the Montgomery potential. Either:

				* '2', for a second-order centered formula;
				* '4', for a fourth-order centered formula.

		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		# Keep track of input parameters
		self._grid     = grid
		self._order	   = order
		self._backend  = backend
		self._dtype	   = dtype

		# Call parent's constructor
		super().__init__(**kwargs)

		# Instantiate the class taking care of the lateral boundary conditions
		self._hboundary = HorizontalBoundary.factory(horizontal_boundary_type,
													 grid, self.nb)

		# Initialize the pointer to the underlying GT4Py stencil calculating
		# the pressure gradient, i.e., the anti-gradient of the Montgomery
		# potential; it will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density':
				{'dims': dims, 'units': 'kg m^-2 K^-1'},
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'x_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic':
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		"""
		Returns
		-------
		int :
			Number of halo layers in the horizontal directions.
		"""
		if self._order == 2:
			return 1
		elif self._order == 4:
			return 2
		else:
			import warnings
			warnings.warn('Order {} not supported; set order to 2.'.format(self._order))
			self._order = 2
			return 1

	@property
	def _stencil_defs(self):
		if self._order == 2:
			return self._stencil_second_order_defs
		elif self._order == 4:
			return self._stencil_fourth_order_defs

	def array_call(self, state):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Instantiate the GT4Py stencil calculating the pressure gradient
		if self._stencil is None:
			self._stencil_initialize()

		# Extract the isentropic density and the Montgomery potential from
		# the input state
		s   = state['air_isentropic_density']
		mtg = state['montgomery_potential']

		# Extend/shrink the isentropic density and the Montgomery potential
		# to accommodate for the lateral boundary conditions
		self._in_s[...]   = self._hboundary.from_physical_to_computational_domain(s)
		self._in_mtg[...] = self._hboundary.from_physical_to_computational_domain(mtg)

		# Run the stencil
		self._stencil.compute()

		# Bring the stencil's outputs back to the physical domain shape.
		# Note that we do not enforce the boundary conditions on the
		# Montgomery potential. Therefore, in the case of relaxed boundary
		# conditions, the outermost halo layers in the output fields would
		# be zero. Nevertheless, periodic conditions are applied exactly.
		su_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_su_tnd, (nx, ny, nz))
		sv_tnd = self._hboundary.from_computational_to_physical_domain(
			self._out_sv_tnd, (nx, ny, nz))

		# Set the other return dictionary
		tendencies = {'x_momentum_isentropic': su_tnd,
					  'y_momentum_isentropic': sv_tnd}

		return tendencies, {}

	def _stencil_initialize(self):
		# Shortcuts
		mi, mj, mk = self._hboundary.mi, self._hboundary.mj, self._grid.nz
		nb = self.nb

		# Allocate the NumPy arrays which serve as stencil's input
		self._in_s   = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._in_mtg = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Allocate the NumPy arrays which serve as stencil's output
		self._out_su_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)
		self._out_sv_tnd = np.zeros((mi, mj, mk), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_s': self._in_s, 'in_mtg': self._in_mtg},
			outputs			 = {'out_su_tnd': self._out_su_tnd,
								'out_sv_tnd': self._out_sv_tnd},
			domain			 = gt.domain.Rectangle((nb, nb, 0),
												   (mi-nb-1, mj-nb-1, mk-1)),
			mode			 = self._backend
		)

	def _stencil_second_order_defs(self, in_s, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# Define the computations
		out_su_tnd[i, j] = in_s[i, j] * (- in_mtg[i+1, j] + in_mtg[i-1, j]) / (2. * dx)
		out_sv_tnd[i, j] = in_s[i, j] * (- in_mtg[i, j+1] + in_mtg[i, j-1]) / (2. * dy)

		return out_su_tnd, out_sv_tnd

	def _stencil_fourth_order_defs(self, in_s, in_mtg):
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# Instantiate the output fields
		out_su_tnd = gt.Equation()
		out_sv_tnd = gt.Equation()

		# Define the computations
		out_su_tnd[i, j] = in_s[i, j] * (- in_mtg[i-2, j]
										 + 8. * in_mtg[i-1, j]
										 - 8. * in_mtg[i+1, j]
										 + in_mtg[i+2, j]) / (12. * dx)
		out_sv_tnd[i, j] = in_s[i, j] * (- in_mtg[i, j-2]
										 + 8. * in_mtg[i, j-1]
										 - 8. * in_mtg[i, j+1]
										 + in_mtg[i, j+2]) / (12. * dy)

		return out_su_tnd, out_sv_tnd


class VerticalIsentropicAdvection(TendencyComponent):
	"""
	This class inherits :class:`sympl.TendencyComponent` to calculate
	the vertical derivative of the conservative vertical advection flux
	in isentropic coordinates for any prognostic variable included in
	the isentropic system.
	"""
	def __init__(self, grid, moist_on=False, flux_scheme='upwind',
				 tendency_of_air_potential_temperature_on_interface_levels=False,
				 backend=gt.mode.NUMPY, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : obj
			TODO
		moist_on : `bool`, optional
			TODO
		flux_scheme : `str`, optional
			TODO
		tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
			TODO
		backend : `obj`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		# Keep track of input arguments
		self._grid     = grid
		self._moist_on = moist_on
		self._stgz     = tendency_of_air_potential_temperature_on_interface_levels
		self._backend  = backend

		# Call parent's constructor
		super().__init__(**kwargs)

		# Instantiate the object calculating the flux
		self._vflux = VerticalHomogeneousIsentropicFlux.factory(flux_scheme, grid, moist_on)

		# Initialize the pointer to the underlying GT4Py stencil;
		# this will be properly redirected the first time the call
		# operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._stgz:
			dims_stgz = (grid.x.dims[0], grid.y.dims[0],
						 grid.z_on_interface_levels.dims[0])
			return_dict['tendency_of_air_potential_temperature_on_interface_levels'] = \
				{'dims': dims_stgz, 'units': 'K s^-1'}
		else:
			return_dict['tendency_of_air_potential_temperature'] = \
				{'dims': dims, 'units': 'K s^-1'}

		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}
		if self._moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	@property
	def nb(self):
		return self._vflux.nb

	def array_call(self, state):
		# Shortcuts
		nb = self.nb
		dtype = state['air_isentropic_density'].dtype

		# Instantiate the stencil object
		if self._stencil is None:
			self._stencil_initialize(dtype)

		# Set the stencil's inputs
		self._stencil_set_inputs(state)

		# Run the stencil
		self._stencil.compute()

		# Set lower layers
		self._set_lower_layers()

		# Collect the output arrays in a dictionary
		tendencies = {
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist_on:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv
			tendencies['mass_fraction_of_cloud_liquid_water_in_air'] = self._out_qc
			tendencies['mass_fraction_of_precipitation_water_in_air'] = self._out_qr

		return tendencies, {}

	def _stencil_initialize(self, dtype):
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mk = nz + 1 if self._stgz else nz
		nb = self.nb

		# Allocate arrays serving as stencil's inputs
		self._in_theta = np.zeros((nx, ny, mk), dtype=dtype)
		self._in_s     = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su    = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv    = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist_on:
			self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate arrays serving as stencil's outputs
		self._out_s  = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist_on:
			self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# Set stencil's inputs
		inputs = {
			'in_theta': self._in_theta, 'in_s': self._in_s,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		if self._moist_on:
			inputs['in_qv'] = self._in_qv
			inputs['in_qc'] = self._in_qc
			inputs['in_qr'] = self._in_qr

		# Set stencil's outputs
		outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv,
		}
		if self._moist_on:
			outputs['out_qv'] = self._out_qv
			outputs['out_qc'] = self._out_qc
			outputs['out_qr'] = self._out_qr

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs 			 = inputs,
			outputs 		 = outputs,
			domain 			 = gt.domain.Rectangle((0, 0, nb), (nx-1, ny-1, nz-nb-1)),
			mode	 		 = self._backend
		)

	def _stencil_set_inputs(self, state):
		if self._stgz:
			self._in_theta[...] = \
				state['tendency_of_air_potential_temperature_on_interface_levels'][...]
		else:
			self._in_theta[...] = state['tendency_of_air_potential_temperature'][...]

		self._in_s[...] 	= state['air_isentropic_density'][...]
		self._in_su[...]    = state['x_momentum_isentropic'][...]
		self._in_sv[...]    = state['y_momentum_isentropic'][...]

		if self._moist_on:
			self._in_qv[...] = state['mass_fraction_of_water_vapor_in_air'][...]
			self._in_qc[...] = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
			self._in_qr[...] = state['mass_fraction_of_precipitation_water_in_air'][...]

	def _stencil_defs(self, in_theta, in_s, in_su, in_sv,
					  in_qv=None, in_qc=None, in_qr=None):
		# Shortcuts
		dz = self._grid.dz.to_units('K').values.item()

		# Indices
		k = gt.Index(axis=2)

		# Output fields
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_qv = gt.Equation()
			out_qc = gt.Equation()
			out_qr = gt.Equation()

		# Vertical velocity
		if self._stgz:
			w = in_theta
		else:
			w = gt.Equation()
			w[k] = 0.5 * (in_theta[k] + in_theta[k-1])

		# Vertical fluxes
		if not self._moist_on:
			flux_s, flux_su, flux_sv = self._vflux(k, w, in_s, in_su, in_sv)
		else:
			flux_s, flux_su, flux_sv, flux_qv, flux_qc, flux_qr = \
				self._vflux(k, w, in_s, in_su, in_sv, in_qv, in_qc, in_qr)

		# Vertical advection
		out_s[k]  = (flux_s[k+1]  - flux_s[k] ) / dz
		out_su[k] = (flux_su[k+1] - flux_su[k]) / dz
		out_sv[k] = (flux_sv[k+1] - flux_sv[k]) / dz
		if self._moist_on:
			out_qv[k] = (flux_qv[k+1] - flux_qv[k]) / dz
			out_qc[k] = (flux_qc[k+1] - flux_qc[k]) / dz
			out_qr[k] = (flux_qr[k+1] - flux_qr[k]) / dz

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_qv, out_qc, out_qr

	def _set_lower_layers(self):
		dz = self._grid.dz.to_units('K').values.item()
		nb = self.nb
		w  = self._in_theta if not self._stgz \
			 else 0.5 * (self._in_theta[:, :, -nb-2:] + self._in_theta[:, :, -nb-3:-1])

		if self._vflux.order == 1:
			self._out_s[:, :, -nb:] = \
				(w[:, :, -nb-1:-1] * self._in_s[:, :, -nb-1:-1] -
				 w[:, :, -nb:    ] * self._in_s[:, :, -nb:    ]) / dz
			self._out_su[:, :, -nb:] = \
				(w[:, :, -nb-1:-1] * self._in_su[:, :, -nb-1:-1] -
				 w[:, :, -nb:    ] * self._in_su[:, :, -nb:    ]) / dz
			self._out_sv[:, :, -nb:] = \
				(w[:, :, -nb-1:-1] * self._in_sv[:, :, -nb-1:-1] -
				 w[:, :, -nb:    ] * self._in_sv[:, :, -nb:    ]) / dz

			if self._moist_on:
				self._out_qv[:, :, -nb:] = \
					(w[:, :, -nb-1:-1] * self._in_qv[:, :, -nb-1:-1] -
					 w[:, :, -nb:    ] * self._in_qv[:, :, -nb:    ]) / dz
				self._out_qc[:, :, -nb:] = \
					(w[:, :, -nb-1:-1] * self._in_qc[:, :, -nb-1:-1] -
					 w[:, :, -nb:    ] * self._in_qc[:, :, -nb:    ]) / dz
				self._out_qr[:, :, -nb:] = \
					(w[:, :, -nb-1:-1] * self._in_qr[:, :, -nb-1:-1] -
					 w[:, :, -nb:    ] * self._in_qr[:, :, -nb:    ]) / dz
		else:
			self._out_s[:, :, -nb:] = \
				0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_s[:, :, -nb:    ]
				 	   + 4.0 * w[:, :, -nb-1:-1] * self._in_s[:, :, -nb-1:-1]
				 	   - 1.0 * w[:, :, -nb-2:-2] * self._in_s[:, :, -nb-2:-2]) / dz
			self._out_su[:, :, -nb:] = \
				0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_su[:, :, -nb:    ]
					   + 4.0 * w[:, :, -nb-1:-1] * self._in_su[:, :, -nb-1:-1]
					   - 1.0 * w[:, :, -nb-2:-2] * self._in_su[:, :, -nb-2:-2]) / dz
			self._out_sv[:, :, -nb:] = \
				0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_sv[:, :, -nb:    ]
					   + 4.0 * w[:, :, -nb-1:-1] * self._in_sv[:, :, -nb-1:-1]
					   - 1.0 * w[:, :, -nb-2:-2] * self._in_sv[:, :, -nb-2:-2]) / dz

			if self._moist_on:
				self._out_qv[:, :, -nb:] = \
					0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_qv[:, :, -nb:    ]
						   + 4.0 * w[:, :, -nb-1:-1] * self._in_qv[:, :, -nb-1:-1]
						   - 1.0 * w[:, :, -nb-2:-2] * self._in_qv[:, :, -nb-2:-2]) / dz
				self._out_qc[:, :, -nb:] = \
					0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_qc[:, :, -nb:    ]
						   + 4.0 * w[:, :, -nb-1:-1] * self._in_qc[:, :, -nb-1:-1]
						   - 1.0 * w[:, :, -nb-2:-2] * self._in_qc[:, :, -nb-2:-2]) / dz
				self._out_qr[:, :, -nb:] = \
					0.5 * (- 3.0 * w[:, :, -nb:    ] * self._in_qr[:, :, -nb:    ]
						   + 4.0 * w[:, :, -nb-1:-1] * self._in_qr[:, :, -nb-1:-1]
						   - 1.0 * w[:, :, -nb-2:-2] * self._in_qr[:, :, -nb-2:-2]) / dz


class PrescribedSurfaceHeating(TendencyComponent):
	"""
	TODO

	References
	----------
	Reisner, J. M., and P. K. Smolarkiewicz. (1994). \
		Thermally forced low Froude number flow past three-dimensional obstacles. \
		*Journal of Atmospheric Sciences*, *51*(1):117-133.
	"""
	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(self, grid, tendency_of_air_potential_temperature_in_diagnostics=False,
				 air_pressure_on_interface_levels=True,
				 amplitude_during_daytime=None, amplitude_at_night=None,
				 attenuation_coefficient_during_daytime=None,
				 attenuation_coefficient_at_night=None,
				 characteristic_length=None, frequency=None, starting_time=None,
				 backend=gt.mode.NUMPY, physical_constants=None, **kwargs):
		"""
		The constructor.

		Parameters
		----------
		grid : grid
			TODO
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			TODO
		air_pressure_on_interface_levels : `bool`, optional
			TODO
		amplitude_during_daytime : `dataarray_like`, optional
			TODO
		amplitude_at_night : `dataarray_like`, optional
			TODO
		attenuation_coefficient_during_daytime : `dataarray_like`, optional
			TODO
		attenuation_coefficient_at_night : `dataarray_like`, optional
			TODO
		characteristic_length : `dataarray_like`, optional
			TODO
		frequency : `dataarray_like`, optional
			TODO
		starting_time : `datetime`, optional
			TODO
		backend : `obj`, optional
			TODO
		physical_constants : `dict`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		self._grid    = grid
		self._tid	  = tendency_of_air_potential_temperature_in_diagnostics
		self._apil	  = air_pressure_on_interface_levels
		self._backend = backend

		super().__init__(**kwargs)

		self._f0d = amplitude_during_daytime.to_units('W m^-2').values.item() \
					if amplitude_during_daytime is not None else 800.0
		self._f0n = amplitude_at_night.to_units('W m^-2').values.item() \
					if amplitude_at_night is not None else -75.0
		self._ad  = attenuation_coefficient_during_daytime.to_units('m^-1').values.item() \
					if attenuation_coefficient_during_daytime is not None else 1.0/600.0
		self._an  = attenuation_coefficient_at_night.to_units('m^-1').values.item() \
					if attenuation_coefficient_at_night is not None else 1.0/75.0
		self._cl  = characteristic_length.to_units('m').values.item() \
					if characteristic_length is not None else 25000.0
		self._w   = frequency.to_units('h^-1').values.item() \
					if frequency is not None else np.pi/12.0
		self._t0  = starting_time

		pcs = get_physical_constants(self._d_physical_constants, physical_constants)
		self._rd = pcs['gas_constant_of_dry_air']
		self._cp = pcs['specific_heat_of_dry_air_at_constant_pressure']

	@property
	def input_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'height_on_interface_levels': {'dims': dims_stgz, 'units': 'm'},
		}

		if self._apil:
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_stgz, 'units': 'Pa'}
		else:
			return_dict['air_pressure'] = {'dims': dims, 'units': 'Pa'}

		return return_dict

	@property
	def tendency_properties(self):
		g = self._grid

		return_dict = {}

		if not self._tid:
			if self._apil:
				dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
				return_dict['air_potential_temperature_on_interface_levels'] = \
					{'dims': dims, 'units': 'K s^-1'}
			else:
				dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
				return_dict['air_potential_temperature'] = {'dims': dims, 'units': 'K s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self._grid

		return_dict = {}

		if self._tid:
			if self._apil:
				dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
				return_dict['tendency_of_air_potential_temperature_on_interface_levels'] = \
					{'dims': dims, 'units': 'K s^-1'}
			else:
				dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
				return_dict['tendency_of_air_potential_temperature'] = \
					{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	def array_call(self, state):
		g = self._grid
		mi, mj = g.nx, g.ny
		mk = g.nz + 1 if self._apil else g.nz

		t    = state['time']
		dt   = (t - self._t0).total_seconds() / 3600.0 if self._t0 is not None \
			else t.hour

		if dt <= 0.0:
			out = np.zeros((mi, mj, mk), dtype=state['height_on_interface_levels'].dtype)
		else:
			x1d, y1d = g.x.to_units('m').values, g.y.to_units('m').values
			theta1d  = g.z_on_interface_levels.to_units('K').values if self._apil \
				else g.z.to_units('K').values

			x 	  = np.tile(x1d[:, np.newaxis, np.newaxis], (1, mj, mk))
			y 	  = np.tile(y1d[np.newaxis, :, np.newaxis], (mi, 1, mk))
			theta = np.tile(theta1d[np.newaxis, np.newaxis, :], (mi, mj, 1))

			p  = state['air_pressure_on_interface_levels'] if self._apil \
				else state['air_pressure']
			zv = state['height_on_interface_levels']
			z  = zv if self._apil else 0.5 * (zv[:, :, 1:] + zv[:, :, :-1])
			h  = np.repeat(zv[:, :, -1:], mk, axis=2)

			f0 = self._f0d if (8.0 <= t.hour < 20) else self._f0n
			a  = self._ad if (8.0 <= t.hour < 20) else self._an
			cl = self._cl
			w  = self._w

			out = (theta * self._rd * a / (p * self._cp) *
				   f0 * np.exp(- a * (z - h)) * np.sin(w * dt)) * (x**2 + y**2 < cl**2)

		tendencies = {}
		if not self._tid:
			if self._apil:
				tendencies['air_potential_temperature_on_interface_levels'] = out
			else:
				tendencies['air_potential_temperature'] = out

		diagnostics = {}
		if self._tid:
			if self._apil:
				diagnostics['tendency_of_air_potential_temperature_on_interface_levels'] = out
			else:
				diagnostics['tendency_of_air_potential_temperature'] = out

		return tendencies, diagnostics
