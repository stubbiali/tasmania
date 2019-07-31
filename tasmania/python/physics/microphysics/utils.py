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
	Clipping(DiagnosticComponent)
	SedimentationFlux
	_{First, Second}OrderUpwind(SedimentationFlux)
	Sedimentation(ImplicitTendencyComponent)
	Precipitation(ImplicitTendencyComponent)
	KesslerMicrophysics(TendencyComponent)
	KesslerSaturationAdjustment(DiagnosticComponent)
	KesslerFallVelocity(DiagnosticComponent)
	PorzMicrophysics(TendencyComponent)
	PorzRaindropFallVelocity(DiagnosticComponent)
"""
import abc
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.framework.base_components import \
	DiagnosticComponent, ImplicitTendencyComponent, TendencyComponent
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.meteo_utils import \
	goff_gratch_formula, tetens_formula

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class Clipping(DiagnosticComponent):
	"""
	Clipping negative values of water species.
	"""
	def __init__(self, domain, grid_type, water_species_names=None):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		water_species_names : `tuple`, optional
			The names of the water species to clip.
		"""
		self._names = water_species_names
		super().__init__(domain, grid_type)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {}
		for name in self._names:
			return_dict[name] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {}
		for name in self._names:
			return_dict[name] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	def array_call(self, state):
		diagnostics = {}

		for name in self._names:
			q = state[name]
			q[q < 0.0] = 0.0
			diagnostics[name] = q

		return diagnostics


class SedimentationFlux:
	"""
	Abstract base class whose derived classes discretize the
	vertical derivative of the sedimentation flux with different
	orders of accuracy.
	"""
	__metaclass__ = abc.ABCMeta

	# the vertical extent of the stencil
	nb = None

	@staticmethod
	@abc.abstractmethod
	def __call__(k, rho, h_on_interface_levels, qr, vt):
		"""
		Get the vertical derivative of the sedimentation flux.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		k : gridtools.Index
			The index running along the vertical axis.
		rho : gridtools.Equation
			The air density, in units of [kg m^-3].
		h_on_interface_levels : gridtools.Equation
			The geometric height of the model half-levels, in units of [m].
		qr : gridtools.Equation
			The mass fraction of precipitation water in air, in units of [g g^-1].
		vt : gridtools.Equation
			The raindrop fall velocity, in units of [m s^-1].

		Return
		------
		gridtools.Equation :
			The vertical derivative of the sedimentation flux.
		"""

	@staticmethod
	def factory(sedimentation_flux_type):
		"""
		Static method returning an instance of the derived class
		which discretizes the vertical derivative of the
		sedimentation flux with the desired level of accuracy.

		Parameters
		----------
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical
			sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		Return
		------
			Instance of the derived class implementing the desired method.
		"""
		if sedimentation_flux_type == 'first_order_upwind':
			return _FirstOrderUpwind()
		elif sedimentation_flux_type == 'second_order_upwind':
			return _SecondOrderUpwind()
		else:
			raise ValueError(
				'Only first- and second-order upwind methods have been implemented.'
			)


class _FirstOrderUpwind(SedimentationFlux):
	"""
	Implementation of the standard, first-order accurate upwind method
	to discretize the vertical derivative of the sedimentation flux.
	"""
	nb = 1

	def __init__(self):
		super().__init__()

	@staticmethod
	def __call__(k, rho, h_on_interface_levels, qr, vt):
		# interpolate the geometric height at the model main levels
		tmp_h = gt.Equation()
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# calculate the vertical derivative of the sedimentation flux
		out_dfdz = gt.Equation(name='tmp_dfdz')
		out_dfdz[k] = (rho[k-1] * qr[k-1] * vt[k-1] - rho[k] * qr[k] * vt[k]) / \
			(tmp_h[k-1] - tmp_h[k])

		return out_dfdz


class _SecondOrderUpwind(SedimentationFlux):
	"""
	Implementation of the second-order accurate upwind method to discretize
	the vertical derivative of the sedimentation flux.
	"""
	nb = 2

	def __init__(self):
		super().__init__()

	@staticmethod
	def __call__(k, rho, h_on_interface_levels, qr, vt):
		# instantiate temporary and output fields
		tmp_h    = gt.Equation()
		tmp_a    = gt.Equation()
		tmp_b    = gt.Equation()
		tmp_c    = gt.Equation()
		out_dfdz = gt.Equation(name='tmp_dfdz')

		# interpolate the geometric height at the model main levels
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# evaluate the space-dependent coefficients occurring in the
		# second-order, upwind finite difference approximation of the
		# vertical derivative of the flux
		tmp_a[k] = (2. * tmp_h[k] - tmp_h[k-1] - tmp_h[k-2]) / \
			((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k]))
		tmp_b[k] = (tmp_h[k-2] - tmp_h[k]) / \
			((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))
		tmp_c[k] = (tmp_h[k] - tmp_h[k-1]) / \
			((tmp_h[k-2] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))

		# calculate the vertical derivative of the sedimentation flux
		out_dfdz[k] = \
			tmp_a[k] * rho[k  ] * qr[k  ] * vt[k  ] + \
			tmp_b[k] * rho[k-1] * qr[k-1] * vt[k-1] + \
			tmp_c[k] * rho[k-2] * qr[k-2] * vt[k-2]

		return out_dfdz


class Precipitation(ImplicitTendencyComponent):
	"""
	Update the (accumulated) precipitation.
	"""
	_d_physical_constants = {
		'density_of_liquid_water':
			DataArray(1e3, attrs={'units': 'kg m^-3'}),
	}

	def __init__(
		self, domain, grid_type='numerical', backend=gt.mode.NUMPY, dtype=datatype,
		physical_constants=None, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'density_of_liquid_water', in units compatible with [kg m^-3].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.ImplicitTendencyComponent`.
		"""
		super().__init__(domain, grid_type, **kwargs)

		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._rhow = gt.Global(pcs['density_of_liquid_water'])

		nx, ny = self.grid.nx, self.grid.ny
		self._dt = gt.Global()
		self._in_rho = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_qr = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_vt = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_accprec = np.zeros((nx, ny, 1), dtype=dtype)
		self._out_prec = np.zeros((nx, ny, 1), dtype=dtype)
		self._out_accprec = np.zeros((nx, ny, 1), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_qr': self._in_qr, 'in_vt': self._in_vt,
				'in_accprec': self._in_accprec
			},
			global_inputs={'rhow': self._rhow, 'dt': self._dt},
			outputs={'out_prec': self._out_prec, 'out_accprec': self._out_accprec},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, 0)),
			mode=backend
		)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims2d = (g.x.dims[0], g.y.dims[0], g.z.dims[0]+'_at_surface_level') \
			if g.nz > 1 else (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'air_density': {'dims': dims, 'units': 'kg m^-3'},
			'mass_fraction_of_precipitation_water_in_air': {'dims': dims, 'units': 'g g^-1'},
			'raindrop_fall_velocity': {'dims': dims, 'units': 'm s^-1'},
			'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'}
		}

	@property
	def tendency_properties(self):
		return {}

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims2d = (g.x.dims[0], g.y.dims[0], g.z.dims[0]+'_at_surface_level') \
			if g.nz > 1 else (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'precipitation': {'dims': dims2d, 'units': 'mm hr^-1'},
			'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'}
		}

	def array_call(self, state, timestep):
		self._dt.value = timestep.total_seconds()
		self._in_rho[...] = state['air_density'][:, :, -1:]
		self._in_qr[...] = state['mass_fraction_of_precipitation_water_in_air'][:, :, -1:]
		self._in_vt[...] = state['raindrop_fall_velocity'][:, :, -1:]
		self._in_accprec[...] = state['accumulated_precipitation'][...]

		self._stencil.compute()

		tendencies = {}
		diagnostics = {
			'precipitation': self._out_prec,
			'accumulated_precipitation': self._out_accprec
		}

		return tendencies, diagnostics

	@staticmethod
	def _stencil_defs(rhow, dt, in_rho, in_qr, in_vt, in_accprec):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		out_prec = gt.Equation()
		out_accprec = gt.Equation()

		out_prec[i, j] = 3.6e6 * in_rho[i, j] * in_qr[i, j] * in_vt[i, j] / rhow
		out_accprec[i, j] = in_accprec[i, j] + dt * out_prec[i, j] / 3.6e3

		return out_prec, out_accprec


class KesslerMicrophysics(TendencyComponent):
	"""
	The WRF version of the Kessler microphysics scheme.

	Note
	----
	The calculated tendencies do not include the source terms deriving
	from the saturation adjustment.

	References
	----------
	Doms, G., et al. (2015). A description of the nonhydrostatic regional \
		COSMO-model. Part II: Physical parameterization. \
		Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
	Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
		Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
		Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
	"""
	# default values for the physical parameters used in the class
	_d_a  = DataArray(0.001, attrs={'units': 'g g^-1'})
	_d_k1 = DataArray(0.001, attrs={'units': 's^-1'})
	_d_k2 = DataArray(2.2, attrs={'units': 's^-1'})

	# default values for the physical constants used in the class
	_d_physical_constants = {
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gas_constant_of_water_vapor':
			DataArray(461.52, attrs={'units': 'J K^-1 kg^-1'}),
		'latent_heat_of_vaporization_of_water':
			DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
	}

	def __init__(
		self, domain, grid_type='numerical', air_pressure_on_interface_levels=True,
		tendency_of_air_potential_temperature_in_diagnostics=False,
		rain_evaporation=True, autoconversion_threshold=_d_a,
		autoconversion_rate=_d_k1, collection_rate=_d_k2,
		saturation_water_vapor_formula='tetens',
		backend=gt.mode.NUMPY, dtype=datatype,
		physical_constants=None, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			:obj:`True` to include the tendency for the potential
			temperature in the output dictionary collecting the diagnostics,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		rain_evaporation : `bool`, optional
			:obj:`True` if the evaporation of raindrops should be taken
			into account, :obj:`False` otherwise. Defaults to :obj:`True`.
		autoconversion_threshold : `sympl.DataArray`, optional
			Autoconversion threshold, in units compatible with [g g^-1].
		autoconversion_rate : `sympl.DataArray`, optional
			Autoconversion rate, in units compatible with [s^-1].
		collection_rate : `sympl.DataArray`, optional
			Rate of collection, in units compatible with [s^-1].
		saturation_water_vapor_formula : `str`, optional
			The formula giving the saturation water vapor. Available options are:

				* 'tetens' (default) for the Tetens' equation;
				* 'goff_gratch' for the Goff-Gratch equation.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gas_constant_of_water_vapor', in units compatible with \
					[J K^-1 kg^-1];
				* 'latent_heat_of_vaporization_of_water', in units compatible with \
					[J kg^-1].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
		# keep track of input arguments
		self._pttd = tendency_of_air_potential_temperature_in_diagnostics
		self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
		self._rain_evaporation = rain_evaporation
		self._a = autoconversion_threshold.to_units('g g^-1').values.item()
		self._k1 = autoconversion_rate.to_units('s^-1').values.item()
		self._k2 = collection_rate.to_units('s^-1').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type, **kwargs)

		# set physical parameters values
		self._physical_constants = get_physical_constants(
			self._d_physical_constants, physical_constants
		)

		# set the formula calculating the saturation water vapor pressure
		self._swvf = \
			goff_gratch_formula if saturation_water_vapor_formula == 'goff_gratch' \
				else tetens_formula

		# shortcuts
		rd = self._physical_constants['gas_constant_of_dry_air']
		rv = self._physical_constants['gas_constant_of_water_vapor']
		self._beta = rd / rv

		# initialize the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		grid = self.grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
		dims_on_interface_levels = (
			grid.x.dims[0],	grid.y.dims[0],	grid.z_on_interface_levels.dims[0]
		)

		return_dict = {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'air_temperature':
				{'dims': dims, 'units': 'K'},
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		if self._air_pressure_on_interface_levels:
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_on_interface_levels, 'units': 'Pa'}
			return_dict['exner_function_on_interface_levels'] = \
				{'dims': dims_on_interface_levels, 'units': 'J K^-1 kg^-1'}
		else:
			return_dict['air_pressure'] = {'dims': dims, 'units': 'Pa'}
			return_dict['exner_function'] = {'dims': dims, 'units': 'J K^-1 kg^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1 s^-1'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1 s^-1'},
		}

		if self._rain_evaporation:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

			if not self._pttd:
				return_dict['air_potential_temperature'] = \
					{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		if self._rain_evaporation and self._pttd:
			grid = self._grid
			dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
			return {
				'tendency_of_air_potential_temperature':
					{'dims': dims, 'units': 'K s^-1'}
			}
		else:
			return {}

	def array_call(self, state):
		# extract the required model variables
		self._in_rho[...] = state['air_density'][...]
		self._in_qv[...]  = state['mass_fraction_of_water_vapor_in_air'][...]
		self._in_qc[...]  = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		if self._air_pressure_on_interface_levels:
			self._in_p[...]   = state['air_pressure_on_interface_levels'][...]
			self._in_exn[...] = state['exner_function_on_interface_levels'][...]
		else:
			self._in_p[...]   = state['air_pressure'][...]
			self._in_exn[...] = state['exner_function'][...]

		# compute the saturation water vapor pressure
		self._in_ps[...] = self._swvf(state['air_temperature'])

		# run the stencil
		#self._stencil.compute()
		p = 0.5 * (self._in_p[:, :, :-1] + self._in_p[:, :, 1:]) \
			if self._air_pressure_on_interface_levels else self._in_p
		exn = 0.5 * (self._in_exn[:, :, :-1] + self._in_exn[:, :, 1:]) \
			if self._air_pressure_on_interface_levels else self._in_exn
		tmp_rho_gcm3 = 0.001 * self._in_rho
		tmp_p_mbar = 0.01 * p
		tmp_qvs = self._beta * self._in_ps / (p - self._in_ps)
		tmp_ar = self._k1 * (self._in_qc > self._a) * (self._in_qc - self._a)
		tmp_cr = np.zeros_like(self._in_rho, dtype=self._in_rho.dtype)
		k = self._in_qr > 0
		tmp_cr[k] = self._k2 * self._in_qc[k] * (self._in_qr[k] ** 0.875)
		if self._rain_evaporation:
			tmp_c = np.zeros_like(self._in_rho, dtype=self._in_rho.dtype)
			tmp_c[k]  = 1.6 + 124.9 * ((tmp_rho_gcm3[k] * self._in_qr[k]) ** 0.2046)
			tmp_er = np.zeros_like(self._in_rho, dtype=self._in_rho.dtype)
			tmp_er[k] = (1. - self._in_qv[k] / tmp_qvs[k]) * tmp_c[k] * \
						((tmp_rho_gcm3[k] * self._in_qr[k]) ** 0.525) / \
						(tmp_rho_gcm3[k] * (5.4e5 + 2.55e6 / (tmp_p_mbar[k] * tmp_qvs[k])))
		if not self._rain_evaporation:
			self._out_qc_tnd[...] = - (tmp_ar[...] + tmp_cr[...])
			self._out_qr_tnd[...] = tmp_ar[...] + tmp_cr[...]
		else:
			self._out_qv_tnd[...] = tmp_er[...]
			self._out_qc_tnd[...] = - (tmp_ar[...] + tmp_cr[...])
			self._out_qr_tnd[...] = tmp_ar[...] + tmp_cr[...] - tmp_er[...]
		if self._rain_evaporation:
			lhvw = self._physical_constants['latent_heat_of_vaporization_of_water']
			self._out_theta_tnd[...] = - lhvw / exn[...] * tmp_er[...]

		# collect the tendencies
		tendencies = {
			'mass_fraction_of_cloud_liquid_water_in_air': self._out_qc_tnd,
			'mass_fraction_of_precipitation_water_in_air': self._out_qr_tnd,
		}
		if self._rain_evaporation:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv_tnd
			if not self._pttd:
				tendencies['air_potential_temperature'] = self._out_theta_tnd

		# collect the diagnostics
		if self._rain_evaporation and self._pttd:
			diagnostics = {'tendency_of_air_potential_temperature': self._out_theta_tnd}
		else:
			diagnostics = {}

		return tendencies, diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		# allocate the numpy arrays which will serve as stencil inputs
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_p   = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_ps  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		if self._air_pressure_on_interface_levels:
			self._in_p   = np.zeros((nx, ny, nz+1), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=dtype)
		else:
			self._in_p   = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the numpy arrays which will serve as stencil outputs
		self._out_qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		if self._rain_evaporation:
			self._out_qv_tnd    = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_theta_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# set stencil's inputs and outputs
		_inputs  = {
			'in_rho': self._in_rho, 'in_p': self._in_p,
			'in_ps': self._in_ps, 'in_exn': self._in_exn,
			'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr,
		}
		_outputs = {'out_qc_tnd': self._out_qc_tnd, 'out_qr_tnd': self._out_qr_tnd}
		if self._rain_evaporation:
			_outputs['out_qv_tnd']    = self._out_qv_tnd
			_outputs['out_theta_tnd'] = self._out_theta_tnd

		# initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=_inputs,
			outputs=_outputs,
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	def _stencil_defs(self, in_rho, in_p, in_ps, in_exn, in_qv, in_qc, in_qr):
		"""
		Definitions function for the GT4Py stencil calculating
		the cloud microphysics tendencies.

		Parameters
		----------
		in_rho : gridtools.Equation
			The air density, in units of [kg m^-3].
		in_p : gridtools.Equation
			The air pressure, in units of [Pa].
		in_ps : gridtools.Equation
			The saturation water vapor pressure, in units of [Pa].
		in_exn : gridtools.Equation
			The Exner function, in units of [J kg^-1 K^-1].
		in_qv : gridtools.Equation
			The mass fraction of water vapor, in units of [g g^-1].
		in_qc : gridtools.Equation
			The mass fraction of cloud liquid water, in units of [g g^-1].
		in_qr : gridtools.Equation
			The mass fraction of precipitation water, in units of [g g^-1].

		Returns
		-------
		out_qc_tnd : gridtools.Equation
			The tendency of mass fraction of cloud liquid water,
			in units of [g g^-1 s^-1].
		out_qr_tnd : gridtools.Equation
			The tendency of mass fraction of precipitation water,
			in units of [g g^-1 s^-1].
		out_qv_tnd : `gridtools.Equation`, optional
			The tendency of mass fraction of water vapor,
			in units of [g g^-1 s^-1].
		out_theta_tnd : `gridtools.Equation`, optional
			The change over time in air potential temperature,
			in units of [K s^-1].
		"""
		# declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# instantiate the temporary fields
		tmp_p_mbar   = gt.Equation()
		tmp_rho_gcm3 = gt.Equation()
		tmp_qvs      = gt.Equation()
		tmp_ar       = gt.Equation()
		tmp_cr       = gt.Equation()
		if self._air_pressure_on_interface_levels:
			tmp_p 	 = gt.Equation()
			tmp_exn	 = gt.Equation()
		if self._rain_evaporation:
			tmp_c    = gt.Equation()
			tmp_er   = gt.Equation()

		# instantiate the output fields
		out_qc_tnd = gt.Equation()
		out_qr_tnd = gt.Equation()
		if self._rain_evaporation:
			out_qv_tnd    = gt.Equation()
			out_theta_tnd = gt.Equation()

		# interpolate the pressure and the Exner function at the vertical main levels
		if self._air_pressure_on_interface_levels:
			tmp_p[k]   = 0.5 * (in_p[k] + in_p[k+1])
			tmp_exn[k] = 0.5 * (in_exn[k] + in_exn[k+1])

		# the pressure and Exner function at the main levels
		p   = tmp_p if self._air_pressure_on_interface_levels else in_p
		exn = tmp_exn if self._air_pressure_on_interface_levels else in_exn

		# perform units conversion
		tmp_rho_gcm3[k] = 0.001 * in_rho[k]
		tmp_p_mbar[k]   = 0.01 * p[k]

		# compute the saturation mixing ratio of water vapor
		tmp_qvs[k] = self._beta * in_ps[k] / (p[k] - in_ps[k])

		# compute the contribution of autoconversion to rain development
		tmp_ar[k] = self._k1 * (in_qc[k] > self._a) * (in_qc[k] - self._a)

		# compute the contribution of accretion to rain development
		tmp_cr[k] = self._k2 * in_qc[k] * (in_qr[k] ** 0.875)

		if self._rain_evaporation:
			# compute the contribution of evaporation to rain development
			tmp_c[k]  = 1.6 + 124.9 * ((tmp_rho_gcm3[k] * in_qr[k]) ** 0.2046)
			tmp_er[k] = (1. - in_qv[k] / tmp_qvs[k]) * tmp_c[k] * \
						((tmp_rho_gcm3[k] * in_qr[k]) ** 0.525) / \
						(tmp_rho_gcm3[k] * (5.4e5 + 2.55e6 / (tmp_p_mbar[k] * tmp_qvs[k])))

		# calculate the tendencies
		if not self._rain_evaporation:
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k]
		else:
			out_qv_tnd[k] = tmp_er[k]
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k] - tmp_er[k]

		# compute the change over time in potential temperature
		if self._rain_evaporation:
			lhvw = self._physical_constants['latent_heat_of_vaporization_of_water']
			out_theta_tnd[k] = - lhvw / exn[k] * tmp_er[k]

		if not self._rain_evaporation:
			return out_qc_tnd, out_qr_tnd
		else:
			return out_qc_tnd, out_qr_tnd, out_qv_tnd, out_theta_tnd


class KesslerSaturationAdjustment(DiagnosticComponent):
	"""
	The saturation adjustment as predicted by the WRF implementation
	of the Kessler microphysics scheme.

	References
	----------
	Doms, G., et al. (2015). A description of the nonhydrostatic regional \
		COSMO-model. Part II: Physical parameterization. \
		Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
	Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
		Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
		Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
	"""
	# default values for the physical constants used in the class
	_d_physical_constants = {
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gas_constant_of_water_vapor':
			DataArray(461.52, attrs={'units': 'J K^-1 kg^-1'}),
		'latent_heat_of_vaporization_of_water':
			DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(
		self, domain, grid_type='numerical', air_pressure_on_interface_levels=True,
		backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gas_constant_of_water_vapor', in units compatible with \
					[J K^-1 kg^-1];
				* 'latent_heat_of_vaporization_of_water', in units compatible with \
					[J kg^-1];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].
		"""
		# keep track of input arguments
		self._apoil = air_pressure_on_interface_levels

		# call parent's constructor
		super().__init__(domain, grid_type)

		# set physical parameters values
		self._physical_constants = get_physical_constants(
			self._d_physical_constants, physical_constants
		)

		# shortcuts
		rd = self._physical_constants['gas_constant_of_dry_air']
		rv = self._physical_constants['gas_constant_of_water_vapor']
		self._beta = rd / rv
		self._lhwv = self._physical_constants['latent_heat_of_vaporization_of_water']
		self._cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

		# initialize the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		grid = self.grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
		dims_on_interface_levels = (
			grid.x.dims[0],	grid.y.dims[0],	grid.z_on_interface_levels.dims[0]
		)

		return_dict = {
			'air_temperature':
				{'dims': dims, 'units': 'K'},
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		if self._apoil:
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_on_interface_levels, 'units': 'Pa'}
		else:
			return_dict['air_pressure'] = {'dims': dims, 'units': 'Pa'}

		return return_dict

	@property
	def diagnostic_properties(self):
		grid = self.grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		return return_dict

	def array_call(self, state):
		# extract the required model variables
		self._in_T[...]	  = state['air_temperature'][...]
		self._in_qv[...]  = state['mass_fraction_of_water_vapor_in_air'][...]
		self._in_qc[...]  = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
		if self._apoil:
			self._in_p[...] = state['air_pressure_on_interface_levels'][...]
		else:
			self._in_p[...] = state['air_pressure'][...]

		# compute the saturation water vapor pressure
		self._in_ps[...] = tetens_formula(self._in_T)

		# run the stencil
		self._stencil.compute()

		# collect the diagnostics
		diagnostics = {
			'mass_fraction_of_water_vapor_in_air': self._out_qv,
			'mass_fraction_of_cloud_liquid_water_in_air': self._out_qc,
		}

		return diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		# allocate the numpy arrays which will serve as stencil inputs
		self._in_ps = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_T  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
		if self._apoil:
			self._in_p = np.zeros((nx, ny, nz+1), dtype=dtype)
		else:
			self._in_p = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the numpy arrays which will serve as stencil outputs
		self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)

		# initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_p': self._in_p, 'in_ps': self._in_ps, 'in_T': self._in_T,
				'in_qv': self._in_qv, 'in_qc': self._in_qc
			},
			outputs={'out_qv': self._out_qv, 'out_qc': self._out_qc},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend)

	def _stencil_defs(self, in_p, in_ps, in_T, in_qv, in_qc):
		"""
		Definitions function for the GT4Py stencil carrying out
		the saturation adjustment.

		Parameters
		----------
		in_p : gridtools.Equation
			The air pressure, in units of [Pa].
		in_ps : gridtools.Equation
			The saturation water vapor pressure, in units of [Pa].
		in_T : gridtools.Equation
			The air temperature, in units of [K].
		in_qv : gridtools.Equation
			The mass fraction of water vapor, in units of [g g^-1].
		in_qc : gridtools.Equation
			The mass fraction of cloud liquid water, in units of [g g^-1].

		Returns
		-------
		out_qv : gridtools.Equation
			The adjusted mass fraction of water vapor, in units of [g g^-1].
		out_qc : gridtools.Equation
			The adjusted mass fraction of cloud liquid water, in units of [g g^-1].
		"""
		# declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# instantiate the temporary fields
		tmp_qvs = gt.Equation()
		tmp_sat = gt.Equation()
		tmp_dlt = gt.Equation()
		if self._apoil:
			tmp_p = gt.Equation()

		# instantiate the output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()

		# interpolate the pressure at the vertical main levels
		if self._apoil:
			tmp_p[k] = 0.5 * (in_p[k] + in_p[k+1])

		# the pressure
		p = tmp_p if self._apoil else in_p

		# compute the saturation mixing ratio of water vapor
		tmp_qvs[k] = self._beta * in_ps[k] / (p[k] - in_ps[k])

		# compute the amount of latent heat released by the condensation of cloud liquid water
		tmp_sat[k] = (tmp_qvs[k] - in_qv[k]) / (
			1. + tmp_qvs[k] * 4093.0 * self._lhwv / (self._cp * (in_T[k] - 36)**2.)
		)

		# compute the source term representing the evaporation of cloud liquid water
		tmp_dlt[k] = (tmp_sat[k] <= in_qc[k]) * tmp_sat[k] + (tmp_sat[k] > in_qc[k]) * in_qc[k]

		# perform the adjustment
		out_qv[k] = in_qv[k] + tmp_dlt[k]
		out_qc[k] = in_qc[k] - tmp_dlt[k]

		return out_qv, out_qc


class KesslerFallVelocity(DiagnosticComponent):
	"""
	Calculate the raindrop fall velocity as prescribed by the Kessler
	microphysics scheme.

	References
	----------
	Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
		Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
		Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
	"""
	def __init__(self, domain, grid_type='numerical', backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		super().__init__(domain, grid_type)

		# initialize the pointer to the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'raindrop_fall_velocity':
				{'dims': dims, 'units': 'm s^-1'},
		}

		return return_dict

	def array_call(self, state):
		# extract the needed model variables
		self._in_rho[...] = state['air_density'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]

		# extract the surface density
		rho_s = self._in_rho[:, :, -1:]
		self._in_rho_s[...] = np.repeat(rho_s, self._grid.nz, axis=2)

		# call the stencil's compute function
		self._stencil.compute()

		# collect the diagnostics
		diagnostics = {
			'raindrop_fall_velocity': self._out_vt,
		}

		return diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		# allocate the numpy arrays which will serve as stencil inputs
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_rho_s = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the numpy array which will serve as stencil output
		self._out_vt = np.zeros((nx, ny, nz), dtype=dtype)

		# initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s, 'in_qr': self._in_qr
			},
			outputs={'out_vt': self._out_vt},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	@staticmethod
	def _stencil_defs(in_rho, in_rho_s, in_qr):
		"""
		Definitions function for the GT4Py stencil calculating the raindrop velocity.

		Parameters
		----------
		in_rho : gridtools.Equation
			The air density, in units of [kg m^-3].
		in_rho_s : gridtools.Equation
			The surface air density, in units of [kg m^-3].
		in_qr : gridtools.Equation
			The mass fraction of precipitation water, in units of [g g^-1].

		Returns
		-------
		gridtools.Equation :
			The raindrop fall velocity, in units of [m s^-1].
		"""
		# declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# instantiate the output field
		out_vt = gt.Equation()

		# perform computations
		out_vt[k] = 36.34 * (1.e-3 * in_rho[k] * (in_qr[k] > 0.) * in_qr[k])**0.1346 * \
					(in_rho_s[k] / in_rho[k])**0.5

		return out_vt


class KesslerSedimentation(ImplicitTendencyComponent):
	"""
	Calculate the vertical derivative of the sedimentation flux for the mass
	fraction of precipitation water.
	"""
	def __init__(
		self, domain, grid_type='numerical',
		sedimentation_flux_scheme='first_order_upwind', maximum_vertical_cfl=0.975,
		backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		sedimentation_flux_scheme : `str`, optional
			The numerical sedimentation flux scheme. Please refer to
			:class:`~tasmania.SedimentationFlux` for the available options.
			Defaults to 'first_order_upwind'.
		maximum_vertical_cfl : `float`, optional
			Maximum allowed vertical CFL number. Defaults to 0.975.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.ImplicitTendencyComponent`.
		"""
		super().__init__(domain, grid_type, **kwargs)
		self._max_cfl = maximum_vertical_cfl
		self._sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'height_on_interface_levels':
				{'dims': dims_z, 'units': 'm'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'raindrop_fall_velocity':
				{'dims': dims, 'units': 'm s^-1'},
		}

	@property
	def tendency_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1 s^-1'}
		}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state, timestep):
		self._stencil_set_inputs(state, timestep)

		self._stencil.compute()

		dh = self._in_h[:, :, :-1] - self._in_h[:, :, 1:]
		x = np.where(self._in_vt > self._max_cfl * dh / timestep.total_seconds())
		if x[0].size > 0:
			print('Number of gps violating vertical CFL: {:4d}'.format(x[0].size))

		tendencies = {'mass_fraction_of_precipitation_water_in_air': self._out_qr}
		diagnostics = {}

		return tendencies, diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		self._dt = gt.Global()
		self._maxcfl = gt.Global(self._max_cfl)

		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_h   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_vt  = np.zeros((nx, ny, nz), dtype=dtype)

		self._out_qr   = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_prec = np.zeros((nx, ny, 1), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_h': self._in_h,
				'in_qr': self._in_qr, 'in_vt': self._in_vt
			},
			global_inputs={'dt': self._dt, 'max_cfl': self._maxcfl},
			outputs={'out_qr': self._out_qr},
			domain=gt.domain.Rectangle((0, 0, self._sflux.nb), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	def _stencil_set_inputs(self, state, timestep):
		self._dt.value = timestep.total_seconds()
		self._in_rho[...] = state['air_density'][...]
		self._in_h[...]   = state['height_on_interface_levels'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		self._in_vt[...]  = state['raindrop_fall_velocity'][...]

	def _stencil_defs(self, dt, max_cfl, in_rho, in_h, in_qr, in_vt):
		k = gt.Index(axis=2)

		tmp_dh = gt.Equation()
		tmp_vt = gt.Equation()

		out_qr = gt.Equation()

		tmp_dh[k] = in_h[k] - in_h[k+1]
		tmp_vt[k] = in_vt[k]
		#(in_vt[k] >  max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
		#(in_vt[k] <= max_cfl * tmp_dh[k] / dt) * in_vt[k]

		tmp_dfdz = self._sflux(k, in_rho, in_h, in_qr, tmp_vt)

		out_qr[k] = tmp_dfdz[k] / in_rho[k]

		return out_qr


class PorzMicrophysics(TendencyComponent):
	"""
	The microphysics scheme proposed by Porz et al. (2018).

	References
	----------
	Porz, N., Hanke, M., Baumgartner, M., and Spichtinger, P. (2018). \
		A model for warm clouds with implicit droplet activation, \
		avoiding saturation adjustment. *Math. Clim. Weather Forecast*, 4:50-78.
	"""
	# useful coefficients
	ae       = 0.78
	alpha    = 190.3
	ak       = 0.002646
	av       = 0.78
	beta     = 4.0/15.0
	bk       = 245.4
	bv       = 0.308
	ck       = -12.0
	D0       = 2.11e-5
	eps      = 0.622
	k1       = 0.0041
	k2       = 0.8
	m0       = 4.0 / 3.0 * np.pi * 1000 * 0.5e-6**3
	mt       = 1.21e-5
	mu0      = 1.458e-6
	N0       = 1000.0
	p_star   = 101325.0
	rho_star = 1.225
	t0       = 273.15
	t_mu     = 110.4
	t_star   = 288.0

	# default value for the activation parameter
	d_ninf = DataArray(8e8, attrs={'units': 'kg^-1'})

	# default values for the physical constants used in the class
	_d_physical_constants = {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'density_of_liquid_water':
			DataArray(1e3, attrs={'units': 'kg m^-3'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gas_constant_of_water_vapor':
			DataArray(461.52, attrs={'units': 'J K^-1 kg^-1'}),
		'latent_heat_of_vaporization_of_water':
			DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'})
	}

	def __init__(
		self, domain, grid_type='numerical', air_pressure_on_interface_levels=True,
		tendency_of_air_potential_temperature_in_diagnostics=False,
		rain_evaporation=True, activation_parameter=d_ninf,
		saturation_water_vapor_formula='tetens', backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			:obj:`True` to include the tendency for the potential
			temperature in the output dictionary collecting the diagnostics,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		rain_evaporation : `bool`, optional
			:obj:`True` if the evaporation of raindrops should be taken
			into account, :obj:`False` otherwise. Defaults to :obj:`True`.
		activation_parameter : `sympl.DataArray`, optional
			The free parameter appearing in Eq. (22) of Porz et al. (2018);
			in units compatible with [kg^-1].
		saturation_water_vapor_formula : `str`, optional
			The formula giving the saturation water vapor. Available options are:

				* 'tetens' (default) for the Tetens' equation;
				* 'goff_gratch' for the Goff-Gratch equation.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'density_of_liquid_water', in units compatible with [kg m^-3];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gas_constant_of_water_vapor', in units compatible with \
					[J K^-1 kg^-1];
				* 'latent_heat_of_vaporization_of_water', in units compatible with \
					[J kg^-1];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units \
					compatible with [J K^-1 kg^-1].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
		# keep track of input arguments
		self._pttd = tendency_of_air_potential_temperature_in_diagnostics
		self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
		self._rain_evaporation = rain_evaporation
		self._ninf = activation_parameter.to_units('kg^-1').values.item()

		# call parent's constructor
		super().__init__(domain, grid_type, **kwargs)

		# set physical parameters values
		self._physical_constants = get_physical_constants(
			self._d_physical_constants, physical_constants
		)

		# set the formula calculating the saturation water vapor pressure
		self._swvf = \
			goff_gratch_formula if saturation_water_vapor_formula == 'goff_gratch' \
			else tetens_formula

		# instantiate the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'air_temperature':
				{'dims': dims, 'units': 'K'},
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'kg kg^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'kg kg^-1'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'kg kg^-1'},
			'number_density_of_precipitation_water':
				{'dims': dims, 'units': 'kg^-1'}
		}

		if self._air_pressure_on_interface_levels:
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_stgz, 'units': 'Pa'}
		else:
			return_dict['air_pressure'] = {'dims': dims, 'units': 'Pa'}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'kg kg^-1 s^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'kg kg^-1 s^-1'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'kg kg^-1 s^-1'},
			'number_density_of_precipitation_water':
				{'dims': dims, 'units': 'kg^-1 s^-1'}
		}

		if not self._pttd:
			return_dict['air_potential_temperature'] = \
				{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		if self._pttd:
			grid = self._grid
			dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
			return {
				'tendency_of_air_potential_temperature':
					{'dims': dims, 'units': 'K s^-1'}
			}
		else:
			return {}

	def array_call(self, state):
		# retrieve needed quantities from input state
		self._in_rho[...] = state['air_density'][...]
		self._in_p[...]   = \
			state['air_pressure'][...] if not self._air_pressure_on_interface_levels \
			else state['air_pressure_on_interface_levels'][...]
		self._in_t[...]   = state['air_temperature'][...]
		self._in_qv[...]  = state['mass_fraction_of_water_vapor_in_air'][...]
		self._in_qc[...]  = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		self._in_nr[...]  = state['number_density_of_precipitation_water'][...]

		# calculate the saturation concentration for water vapor
		self._in_ps[...] = self._swvf(self._in_t)

		# calculate the number density of cloud droplets
		qc, Ninf = self._in_qc, self._ninf
		self._in_nc[...] = qc[...] * Ninf / \
			((qc[...] + Ninf * self.m0) * np.tanh(qc[...] / (self.N0 * self.m0)))

		if False:
			# evaluate the stencil
			self._stencil.compute()
		else:
			# shortcuts
			pref = self._physical_constants['air_pressure_at_sea_level']
			rhol = self._physical_constants['density_of_liquid_water']
			rd   = self._physical_constants['gas_constant_of_dry_air']
			rv   = self._physical_constants['gas_constant_of_water_vapor']
			l    = self._physical_constants['latent_heat_of_vaporization_of_water']
			cp   = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
			rho  = self._in_rho
			p    = self._in_p if not self._air_pressure_on_interface_levels \
				else 0.5 * (self._in_p[:, :, :-1] + self._in_p[:, :, 1:])
			ps   = self._in_ps
			t    = self._in_t
			qv   = self._in_qv
			qc   = self._in_qc
			qr   = self._in_qr
			nc   = self._in_nc
			nr   = self._in_nr

			# calculate some of the coefficients needed to compute the tendencies
			D  = self.D0 * (t / self.t0)**1.94 * self.p_star / p
			K  = self.ak * t**1.5 / (t + self.bk * 10**(self.ck / t))
			G  = 1.0 / ((l / (rv * t) - 1.0) * l * ps * D / (rv * K * t**2) + 1.0)
			d  = 4.0 * np.pi * (3.0 / (4.0 * np.pi * rhol))**(1.0 / 3.0) * D * G
			mu = self.mu0 * t**1.5 / (t + self.t_mu)
			be = self.bv * (mu / (rho * D))**(1.0 / 3.0) * (2.0 * rho / mu)**0.5 * \
			 	(3.0 / (4.0 * np.pi * rhol))**(1.0 / 6.0)

			# calculate the terminal velocity of water particles
			vt = self.alpha * qr**self.beta * (self.mt / (qr + self.mt * nr))**self.beta * \
				(self.rho_star / rho)**0.5

			# calculate the saturation mixing ratio of water vapor
			qvs = self.eps * ps / p

			# calculate the tendencies due to autoconversion, accretion and condensation
			A1  = self.k1 * rho * qc**2 / rhol
			A1p = 0.5 * self.k1 * rho * nc * qc / rhol
			A2  = self.k2 * np.pi * (3.0 / (4.0 * np.pi * rhol))**(2.0 / 3.0) * \
			 	vt * rho * qc * qr**(2.0 / 3.0) * nr**(1.0 / 3.0)
			C   = d * rho * (qv - qvs) * nc**(2.0 / 3.0) * qc**(1.0 / 3.0)

			if self._rain_evaporation:
				# calculate the tendencies due to evaporation
				E  = d * rho * ((qvs - qv) > 0.0) * (qvs - qv) * \
					 (self.ae * qr**(1.0 / 3.0) * nr**(2.0 / 3.0) +
					  be * vt**0.5 * qr**0.5 * nr**0.5)
				Ep = E * nr / qr
				Ep[qr <= 0.0] = 0.0

			# calculate the overall tendencies
			if self._rain_evaporation:
				self._out_qv[...] = - C[...] + E[...]
				self._out_qc[...] = C[...] - A1[...] - A2[...]
				self._out_qr[...] = A1[...] + A2[...] - E[...]
				self._out_nr[...] = A1p[...] - Ep[...]
				self._out_theta[...] = (pref / p[...])**(rd / cp) * l * (C[...] - E[...]) / cp
			else:
				self._out_qv[...] = - C[...]
				self._out_qc[...] = C[...] - A1[...] - A2[...]
				self._out_qr[...] = A1[...] + A2[...]
				self._out_nr[...] = A1p[...]
				self._out_theta[...] = (pref / p[...])**(rd / cp) * l * C[...] / cp

		# set the outputs
		tendencies = {
			'mass_fraction_of_water_vapor_in_air': self._out_qv,
			'mass_fraction_of_cloud_liquid_water_in_air': self._out_qc,
			'mass_fraction_of_precipitation_water_in_air': self._out_qr,
			'number_density_of_precipitation_water': self._out_nr,
		}
		if self._pttd:
			diagnostics = {'tendency_of_air_potential_temperature': self._out_theta}
		else:
			tendencies['air_potential_temperature'] = self._out_theta
			diagnostics = {}

		return tendencies, diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		# allocate the numpy arrays serving as stencil inputs
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_p   = np.zeros(
			(nx, ny, nz+1 if self._air_pressure_on_interface_levels else nz),
			dtype=dtype
		)
		self._in_ps  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_t   = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_nc  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_nr  = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the numpy arrays serving as stencil outputs
		self._out_qv    = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qc    = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr    = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_nr    = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_theta = np.zeros((nx, ny, nz), dtype=dtype)

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_p': self._in_p, 'in_ps': self._in_ps,
				'in_t': self._in_t, 'in_qv': self._in_qv, 'in_qc': self._in_qc,
				'in_qr': self._in_qr, 'in_nc': self._in_nc, 'in_nr': self._in_nr
			},
			outputs={
				'out_qv': self._out_qv, 'out_qc': self._out_qc, 'out_qr': self._out_qr,
				'out_nr': self._out_nr, 'out_theta': self._out_theta
			},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	def _stencil_defs(
		self, in_rho, in_p, in_ps, in_t, in_qv, in_qc, in_qr, in_nc, in_nr
	):
		# shortcuts
		pref = self._physical_constants['air_pressure_at_sea_level']
		rhol = self._physical_constants['density_of_liquid_water']
		rd   = self._physical_constants['gas_constant_of_dry_air']
		rw   = self._physical_constants['gas_constant_of_water_vapor']
		l    = self._physical_constants['latent_heat_of_vaporization_of_water']
		cp   = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

		# instantiate the indices
		k = gt.Index(axis=2)

		# allocate the temporary fields
		p   = gt.Equation()
		qvs = gt.Equation()
		D   = gt.Equation()
		K   = gt.Equation()
		G   = gt.Equation()
		d   = gt.Equation()
		mu  = gt.Equation()
		be  = gt.Equation()
		vt  = gt.Equation()
		A1  = gt.Equation()
		A1p = gt.Equation()
		A2  = gt.Equation()
		C   = gt.Equation()
		if self._rain_evaporation:
			E  = gt.Equation()
			Ep = gt.Equation()

		# allocate the output fields
		out_qv 	  = gt.Equation()
		out_qc 	  = gt.Equation()
		out_qr 	  = gt.Equation()
		out_nr 	  = gt.Equation()
		out_theta = gt.Equation()

		# computations
		## TODO


class PorzFallVelocity(DiagnosticComponent):
	"""
	Calculate the effective fall velocity for the mass and number
	concentration of raindrops as prescribed by the microphysics scheme
	proposed by Porz et al. (2018).

	References
	----------
	Porz, N., Hanke, M., Baumgartner, M., and Spichtinger, P. (2018). \
		A model for warm clouds with implicit droplet activation, \
		avoiding saturation adjustment. *Math. Clim. Weather Forecast*, 4:50-78.
	"""
	# useful coefficients
	alpha    = 190.3
	beta     = 4.0/15.0
	cn       = 0.58
	cq       = 1.84
	mt       = 1.21e-5
	rho_star = 1.225

	def __init__(self, domain, grid_type='numerical', backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		super().__init__(domain, grid_type)

		# initialize the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'kg kg^-1'},
			'number_density_of_precipitation_water':
				{'dims': dims, 'units': 'kg^-1'}
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'raindrop_fall_velocity':
				{'dims': dims, 'units': 'm s^-1'},
			'number_density_fall_velocity':
				{'dims': dims, 'units': 'm s^-1'}
		}

		return return_dict

	def array_call(self, state):
		# extract the needed model variables
		self._in_rho[...] = state['air_density'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		self._in_nr[...]  = state['number_density_of_precipitation_water'][...]

		# call the stencil's compute function
		self._stencil.compute()

		# collect the diagnostics
		diagnostics = {
			'raindrop_fall_velocity': self._out_vq,
			'number_density_fall_velocity': self._out_vn
		}

		return diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		# allocate the numpy arrays which will serve as stencil inputs
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_nr  = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the numpy arrays which will serve as stencil outputs
		self._out_vq = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_vn = np.zeros((nx, ny, nz), dtype=dtype)

		# initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_qr': self._in_qr, 'in_nr': self._in_nr
			},
			outputs={'out_vq': self._out_vq, 'out_vn': self._out_vn},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	def _stencil_defs(self, in_rho, in_qr, in_nr):
		# declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# instantiate the temporary and output fields
		vt = gt.Equation()
		out_vq = gt.Equation()
		out_vn = gt.Equation()

		# perform computations
		vt[k] = self.alpha * in_qr[k]**self.beta * \
			(self.mt / (in_qr[k] + self.mt * in_nr[k]))**self.beta * \
			(self.rho_star / in_rho[k])**0.5
		out_vq[k] = self.cq * vt[k]
		out_vn[k] = self.cn * vt[k]

		return out_vq, out_vn


