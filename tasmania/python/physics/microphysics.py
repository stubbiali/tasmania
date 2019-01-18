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
	Kessler(TendencyComponent)
	SaturationAdjustmentKessler(DiagnosticComponent)
	RaindropFallVelocity(DiagnosticComponent)
    SedimentationFlux
    _{First, Second}OrderUpwind(SedimentationFlux)
    Sedimentation(ImplicitTendencyComponent)
"""
import abc
import numpy as np
from sympl import \
	DataArray, DiagnosticComponent, TendencyComponent, ImplicitTendencyComponent

import gridtools as gt
from tasmania.python.utils.data_utils import get_physical_constants


class Kessler(TendencyComponent):
	"""
	This class inherits :class:`sympl.Prognostic` to implement the WRF
	version of the Kessler microphysics scheme.

	Note
	----
	The calculated tendencies do not include the source terms deriving
	from the saturation adjustment.
	"""
	# Default values for the physical parameters used in the class
	_d_a  = DataArray(0.001, attrs={'units': 'g g^-1'})
	_d_k1 = DataArray(0.001, attrs={'units': 's^-1'})
	_d_k2 = DataArray(2.2, attrs={'units': 's^-1'})

	# Default values for the physical constants used in the class
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
		self, grid, air_pressure_on_interface_levels=True,
		tendency_of_air_potential_temperature_in_diagnostics=False,
		rain_evaporation=True, autoconversion_threshold=_d_a,
		rate_of_autoconversion=_d_k1, rate_of_collection=_d_k2,
		backend=gt.mode.NUMPY, physical_constants=None, **kwargs
	):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes.
		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			:obj:`True` to include the tendency for the potential temperature
			in the output dictionary collecting the diagnostics, :obj:`False`
			otherwise. Defaults to :obj:`False`.
		rain_evaporation : `bool`, optional
			:obj:`True` if the evaporation of raindrops should be taken
			into account, :obj:`False` otherwise. Defaults to :obj:`True`.
		autoconversion_threshold : `dataarray_like`, optional
			Autoconversion threshold, in units compatible with [g g^-1].
			Defaults to :attr:`~tasmania.physics.microphysics.Kessler._d_a`.
		rate_of_autoconversion : `dataarray_like`, optional
			Rate of autoconversion, in units compatible with [s^-1].
			Defaults to :attr:`~tasmania.physics.microphysics.Kessler._d_k1`.
		rate_of_collection : `dataarray_like`, optional
			Rate of collection, in units compatible with [s^-1].
			Defaults to :attr:`~tasmania.physics.microphysics.Kessler._d_k2`.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :obj:`gridtools.mode.NUMPY`.
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

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.physics.microphysics.Kessler._d_physical_constants`
			for the default values.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		# Keep track of input arguments
		self._grid = grid
		self._pttd = tendency_of_air_potential_temperature_in_diagnostics
		self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
		self._rain_evaporation = rain_evaporation
		self._a = autoconversion_threshold.to_units('g g^-1').values.item()
		self._k1 = rate_of_autoconversion.to_units('s^-1').values.item()
		self._k2 = rate_of_collection.to_units('s^-1').values.item()
		self._backend = backend

		# Call parent's constructor
		super().__init__(**kwargs)

		# Set physical parameters values
		self._physical_constants = get_physical_constants(self._d_physical_constants,
														  physical_constants)

		# Constants for Tetens' formula
		self._p0    = 610.78
		self._alpha = 17.27
		self._Tr    = 273.15
		self._bw    = 35.85
		
		# Shortcuts
		Rd = self._physical_constants['gas_constant_of_dry_air']
		Rv = self._physical_constants['gas_constant_of_water_vapor']
		L  = self._physical_constants['latent_heat_of_vaporization_of_water']
		cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		self._beta   = Rd / Rv
		self._beta_c = 1. - self._beta
		self._kappa  = L * self._alpha * self._beta * (self._Tr - self._bw) / cp 

		# Initialize the pointer to the underlying GT4Py stencil
		# This will be properly re-directed the first time the call method is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
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
			return {'tendency_of_air_potential_temperature':
						{'dims': dims, 'units': 'K s^-1'}}
		else:
			return {}

	def array_call(self, state):
		"""
		Compute the output cloud microphysical tendencies via a GT4Py stencil.

		Parameters
		----------
        state : dict
            Dictionary whose keys are strings indicating the model
            variables required to calculate the tendencies, and values
            are :class:`numpy.ndarray`\s containing the data for those
            variables.

		Returns
		-------
		tendencies : dict
            Dictionary whose keys are strings indicating the calculated
            tendencies, and values are :class:`numpy.ndarray`\s containing
            the data for those tendencies.
		diagnostics : dict
            Dictionary whose keys are strings indicating the calculated
            diagnostics, and values are :class:`numpy.ndarray`\s containing
            the data for those diagnostics.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(state['air_density'].dtype)

		# Extract the required model variables
		self._in_rho[...] = state['air_density'][...]
		self._in_T[...]	  = state['air_temperature'][...]
		self._in_qv[...]  = state['mass_fraction_of_water_vapor_in_air'][...]
		self._in_qc[...]  = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		if self._air_pressure_on_interface_levels:
			self._in_p[...]   = state['air_pressure_on_interface_levels'][...]
			self._in_exn[...] = state['exner_function_on_interface_levels'][...]
		else:
			self._in_p[...]   = state['air_pressure'][...]
			self._in_exn[...] = state['exner_function'][...]

		# Compute the saturation water vapor pressure via Tetens' formula
		self._in_ps[...] = self._p0 * np.exp(self._alpha *
											 (self._in_T[...] - self._Tr) /
											 (self._in_T[...] - self._bw))

		# Run the stencil
		self._stencil.compute()

		# Collect the tendencies
		tendencies = {
			'mass_fraction_of_cloud_liquid_water_in_air': self._out_qc_tnd,
			'mass_fraction_of_precipitation_water_in_air': self._out_qr_tnd,
		}
		if self._rain_evaporation:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv_tnd
			if not self._pttd:
				tendencies['air_potential_temperature'] = self._out_theta_tnd

		# Collect the diagnostics
		if self._rain_evaporation and self._pttd:
			diagnostics = {'tendency_of_air_potential_temperature': self._out_theta_tnd}
		else:
			diagnostics = {}

		return tendencies, diagnostics

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil in charge of calculating the cloud
		microphysical tendencies.

		Parameters
		----------
		dtype : obj
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
		"""
		# Allocate the numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_p   = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_ps  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_T   = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		if self._air_pressure_on_interface_levels:
			self._in_p   = np.zeros((nx, ny, nz+1), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=dtype)
		else:
			self._in_p   = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the numpy arrays which will serve as stencil outputs
		self._out_qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		if self._rain_evaporation:
			self._out_qv_tnd    = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_theta_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# Set stencil's inputs and outputs
		_inputs  = {
			'in_rho': self._in_rho, 'in_p': self._in_p,
			'in_ps': self._in_ps, 'in_exn': self._in_exn,
			'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr,
		}
		_outputs = {'out_qc_tnd': self._out_qc_tnd, 'out_qr_tnd': self._out_qr_tnd}
		if self._rain_evaporation:
			_outputs['out_qv_tnd']    = self._out_qv_tnd
			_outputs['out_theta_tnd'] = self._out_theta_tnd

		# Initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = _inputs,
			outputs			 = _outputs,
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

	def _stencil_defs(self, in_rho, in_p, in_ps, in_exn, in_qv, in_qc, in_qr):
		"""
		Definitions function for the GT4Py stencil calculating
		the cloud microphysics tendencies.

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_p : obj
			:class:`gridtools.Equation` representing the air pressure.
		in_ps : obj
			:class:`gridtools.Equation` representing the saturation
			vapor pressure.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction
			of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction
			of cloud liquid water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction
			of precipitation water.

		Returns
		-------
		out_qc_tnd : obj
			:class:`gridtools.Equation` representing the tendency of
			mass fraction of cloud liquid water.
		out_qr_tnd : obj
			:class:`gridtools.Equation` representing the tendency of
			mass fraction of precipitation water.
		out_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of
			mass fraction of water vapor.
		out_theta_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the change over time in
			air potential temperature.

		References
		----------
		Doms, G., et al. (2015). A description of the nonhydrostatic regional \
			COSMO-model. Part II: Physical parameterization. \
			Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
		Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
			Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
			Kessler cloud microphysics scheme. *Computer \& Geosciences*, *52*:292-299.
		"""
		# Declare the index scanning along the vertical axis
		k = gt.Index(axis=2)

		# Instantiate the temporary fields
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

		# Instantiate the output fields
		out_qc_tnd = gt.Equation()
		out_qr_tnd = gt.Equation()
		if self._rain_evaporation:
			out_qv_tnd    = gt.Equation()
			out_theta_tnd = gt.Equation()

		# Interpolate the pressure and the Exner function at the vertical main levels
		if self._air_pressure_on_interface_levels:
			tmp_p[k]   = 0.5 * (in_p[k] + in_p[k+1])
			tmp_exn[k] = 0.5 * (in_exn[k] + in_exn[k+1])

		# Set pointers to equations representing pressure and Exner function
		# at the main levels
		p   = tmp_p if self._air_pressure_on_interface_levels else in_p
		exn = tmp_exn if self._air_pressure_on_interface_levels else in_exn

		# Perform units conversion
		tmp_rho_gcm3[k] = 1.e3 * in_rho[k]
		tmp_p_mbar[k]   = 1.e-2 * p[k]

		# Compute the saturation mixing ratio of water vapor
		tmp_qvs[k] = self._beta * in_ps[k] / (p[k] - self._beta_c * in_ps[k])

		# Compute the contribution of autoconversion to rain development
		tmp_ar[k] = self._k1 * (in_qc[k] > self._a) * (in_qc[k] - self._a)

		# Compute the contribution of accretion to rain development
		tmp_cr[k] = self._k2 * in_qc[k] * (in_qr[k] ** 0.875)

		if self._rain_evaporation:
			# Compute the contribution of evaporation to rain development
			tmp_c[k]  = 1.6 + 124.9 * ((tmp_rho_gcm3[k] * in_qr[k]) ** 0.2046)
			tmp_er[k] = (1. - in_qv[k] / tmp_qvs[k]) * tmp_c[k] * \
						((tmp_rho_gcm3[k] * in_qr[k]) ** 0.525) / \
						(tmp_rho_gcm3[k] * (5.4e5 + 2.55e6 / (tmp_p_mbar[k] * tmp_qvs[k])))

		# Calculate the tendencies
		if not self._rain_evaporation:
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k]
		else:
			out_qv_tnd[k] = tmp_er[k]
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k] - tmp_er[k]

		# Compute the change over time in potential temperature
		if self._rain_evaporation:
			lhvw = self._physical_constants['latent_heat_of_vaporization_of_water']
			out_theta_tnd[k] = - lhvw / exn[k] * tmp_er[k]

		if not self._rain_evaporation:
			return out_qc_tnd, out_qr_tnd
		else:
			return out_qc_tnd, out_qr_tnd, out_qv_tnd, out_theta_tnd


class SaturationAdjustmentKessler(DiagnosticComponent):
	"""
	This class inherits :class:`sympl.DiagnosticComponent` to implement the saturation
	adjustment as predicted by the WRF implementation of the Kessler scheme.
	"""
	# Default values for the physical constants used in the class
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

	def __init__(self, grid, air_pressure_on_interface_levels=True,
				 backend=gt.mode.NUMPY, physical_constants=None, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			 or one of its derived classes.
		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :obj:`gridtools.mode.NUMPY`.
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

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.physics.microphysics.SaturationAdjustmentKessler._d_physical_constants`
			for the default values.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.DiagnosticComponent`.
		"""
		# Keep track of input arguments
		self._grid = grid
		self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
		self._backend = backend

		# Call parent's constructor
		super().__init__(**kwargs)

		# Set physical parameters values
		self._physical_constants = get_physical_constants(self._d_physical_constants,
														  physical_constants)

		# Constants for Tetens' formula
		self._p0    = 610.78
		self._alpha = 17.27
		self._Tr    = 273.15
		self._bw    = 35.85

		# Shortcuts
		Rd = self._physical_constants['gas_constant_of_dry_air']
		Rv = self._physical_constants['gas_constant_of_water_vapor']
		L  = self._physical_constants['latent_heat_of_vaporization_of_water']
		cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		self._beta   = Rd / Rv
		self._beta_c = 1. - self._beta
		self._kappa  = L * self._alpha * self._beta * (self._Tr - self._bw) / cp

		# Initialize the pointer to the underlying GT4Py stencil
		# This will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
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

		if self._air_pressure_on_interface_levels:
			return_dict['air_pressure_on_interface_levels'] = \
				{'dims': dims_on_interface_levels, 'units': 'Pa'}
		else:
			return_dict['air_pressure'] = {'dims': dims, 'units': 'Pa'}

		return return_dict

	@property
	def diagnostic_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		return return_dict

	def array_call(self, state):
		"""
		Adjust the distribution of water vapor and cloud liquid water
		via a GT4Py stencil.

		Parameters
		----------
        state : dict
            Dictionary whose keys are strings indicating the model
            variables required to perform the adjustments, and values
            are :class:`numpy.ndarray`\s containing the data for
            those variables.

		Returns
		-------
		diagnostics : dict
            Dictionary whose keys are strings indicating the calculated
            diagnostics, and values are :class:`numpy.ndarray`\s containing
            the data for those diagnostics.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(state['air_temperature'].dtype)

		# Extract the required model variables
		self._in_T[...]	  = state['air_temperature'][...]
		self._in_qv[...]  = state['mass_fraction_of_water_vapor_in_air'][...]
		self._in_qc[...]  = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
		if self._air_pressure_on_interface_levels:
			self._in_p[...] = state['air_pressure_on_interface_levels'][...]
		else:
			self._in_p[...] = state['air_pressure'][...]

		# Compute the saturation water vapor pressure via Tetens' formula
		self._in_ps[...] = self._p0 * np.exp(self._alpha *
											 (self._in_T[...] - self._Tr) /
											 (self._in_T[...] - self._bw))

		# Run the stencil
		self._stencil.compute()

		# Collect the diagnostics
		diagnostics = {
			'mass_fraction_of_water_vapor_in_air': self._out_qv,
			'mass_fraction_of_cloud_liquid_water_in_air': self._out_qc,
		}

		return diagnostics

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil in charge of carrying out
		the saturation adjustment.
		"""
		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_ps = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_T  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
		if self._air_pressure_on_interface_levels:
			self._in_p = np.zeros((nx, ny, nz+1), dtype=dtype)
		else:
			self._in_p = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the Numpy arrays which will serve as stencil outputs
		self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)

		# Initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {'in_p': self._in_p, 'in_ps': self._in_ps, 'in_T': self._in_T,
								'in_qv': self._in_qv, 'in_qc': self._in_qc},
			outputs			 = {'out_qv': self._out_qv, 'out_qc': self._out_qc},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

	def _stencil_defs(self, in_p, in_ps, in_T, in_qv, in_qc):
		"""
		Definitions function for the GT4Py stencil carrying out
		the saturation adjustment.

		Parameters
		----------
		in_p : obj
			:class:`gridtools.Equation` representing the air pressure.
		in_ps : obj
			:class:`gridtools.Equation` representing the saturation
			vapor pressure.
		in_T : obj
			:class:`gridtools.Equation` representing the air temperature.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction
			of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction
			of cloud liquid water.

		Returns
		-------
		out_qv : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction
			of water vapor.
		out_qc : obj
			:class:`gridtools.Equation` representing the adjusted mass fraction
			of cloud liquid water.

		References
		----------
		Doms, G., et al. (2015). *A description of the nonhydrostatic regional \
			COSMO-model. Part II: Physical parameterization.* \
			Retrieved from `COSMO <http://www.cosmo-model.org>`_. \
		Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
			*Compute Unified Device Architecture (CUDA)-based parallelization of WRF \
			Kessler cloud microphysics scheme*. Computer \& Geosciences, 52:292-299.
		"""
		# Declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# Instantiate the temporary fields
		tmp_qvs = gt.Equation()
		tmp_sat = gt.Equation()
		tmp_dlt = gt.Equation()
		if self._air_pressure_on_interface_levels:
			tmp_p = gt.Equation()

		# Instantiate the output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()

		# Interpolate the pressure at the vertical main levels
		if self._air_pressure_on_interface_levels:
			tmp_p[k] = 0.5 * (in_p[k] + in_p[k+1])

		# Set the pointer to the equation representing the pressure
		p = tmp_p if self._air_pressure_on_interface_levels else in_p

		# Compute the saturation mixing ratio of water vapor
		tmp_qvs[k] = self._beta * in_ps[k] / (p[k] - self._beta_c * in_ps[k])

		# Compute the amount of latent heat released by the condensation
		# of cloud liquid water
		tmp_sat[k] = (tmp_qvs[k] - in_qv[k]) / \
					 (1. + self._kappa * in_ps[k] /
					  ((in_T[k] - self._bw)**2. *
					   (in_p[k] - self._beta * in_ps[k])**2.))

		# Compute the source term representing the evaporation of cloud liquid water
		tmp_dlt[k] = (tmp_sat[k] <= in_qc[k]) * tmp_sat[k] + \
					 (tmp_sat[k] > in_qc[k]) * in_qc[k]

		# Perform the adjustment
		out_qv[k] = in_qv[k] + tmp_dlt[k]
		out_qc[k] = in_qc[k] - tmp_dlt[k]

		return out_qv, out_qc


class RaindropFallVelocity(DiagnosticComponent):
	"""
	This class inherits :class:`sympl.DiagnosticComponent` to calculate
	the raindrop fall velocity.
	"""
	def __init__(self, grid, backend=gt.mode.NUMPY, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			 or one of its derived classes.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :obj:`gridtools.mode.NUMPY`.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.DiagnosticComponent`.
		"""
		# Keep track of input arguments
		self._grid = grid
		self._backend = backend

		# Call parent's constructor
		super().__init__(**kwargs)

		# Initialize the pointer to the underlying GT4Py stencil
		# This will be properly re-directed the first time the
		# call operator is invoked
		self._stencil = None

	@property
	def input_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_density':
				{'dims': dims, 'units': 'kg m^-3'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		grid = self._grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'raindrop_fall_velocity':
				{'dims': dims, 'units': 'm s^-1'},
		}

		return return_dict

	def array_call(self, state):
		"""
		Calculate the raindrop fall velocity via a GT4Py stencil.

		Parameters
		----------
        state : dict
            Dictionary whose keys are strings indicating the model
            variables required to perform the adjustments, and values
            are :class:`numpy.ndarray`\s containing the data for
            those variables.

		Returns
		-------
		diagnostics : dict
            Dictionary whose keys are strings indicating the calculated
            diagnostics, and values are :class:`numpy.ndarray`\s containing
            the data for those diagnostics.
		"""
		# If this is the first time this method is invoked,
		# initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(state['air_density'].dtype)

		# Extract the needed model variables
		self._in_rho[...] = state['air_density'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]

		# Extract the surface density
		rho_s = self._in_rho[:, :, -1:]
		self._in_rho_s[...] = np.repeat(rho_s, self._grid.nz, axis=2)

		# Call the stencil's compute function
		self._stencil.compute()

		# Collect the diagnostics
		diagnostics = {
			'raindrop_fall_velocity': self._out_vt,
		}

		return diagnostics

	def _stencil_initialize(self, dtype):
		"""
		Initialize the GT4Py stencil calculating the raindrop velocity.
		"""
		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_rho_s = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the Numpy array which will serve as stencil output
		self._out_vt = np.zeros((nx, ny, nz), dtype=dtype)

		# Initialize the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = {
				'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s, 'in_qr': self._in_qr
			},
			outputs			 = {'out_vt': self._out_vt},
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend)

	@staticmethod
	def _stencil_defs(in_rho, in_rho_s, in_qr):
		"""
		Definitions function for the GT4Py stencil calculating
		the raindrop velocity.

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the density.
		in_rho_s : obj
			:class:`gridtools.Equation` representing the surface density.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction
			of precipitation water.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		"""
		# Declare the index scanning the vertical axis
		k = gt.Index(axis=2)

		# Instantiate the output field
		out_vt = gt.Equation()

		# Perform computations
		out_vt[k] = 36.34 * (1.e-3 * in_rho[k] * (in_qr[k] > 0.) * in_qr[k])**0.1346 * \
					(in_rho_s[k] / in_rho[k])**0.5

		return out_vt


class SedimentationFlux:
	"""
	Abstract base class whose derived classes discretize the
	vertical derivative of the sedimentation flux with different
	orders of accuracy.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		"""
		Get the vertical derivative of the sedimentation flux.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		rho : obj
			:class:`gridtools.Equation` representing the air density.
		h_on_interface_levels : obj
			:class:`gridtools.Equation` representing the geometric
			height of the model half-levels.
		qr : obj
			:class:`gridtools.Equation` representing the mass fraction
			of precipitation water in air.
		vt : obj
			:class:`gridtools.Equation` representing the raindrop
			fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the vertical
			derivative of the sedimentation flux.
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
	This class inherits
	:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
	to implement a standard, first-order accurate upwind method
	to discretize the vertical derivative of the sedimentation flux.

	Attributes
	----------
	nb : int
		Extent of the stencil in the upward vertical direction.
	"""
	def __init__(self):
		"""
		Note
		----
		To instantiate an object of this class, one should prefer
		the static method
		:meth:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux.factory`
		of :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 1

	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		# Interpolate the geometric height at the model main levels
		tmp_h = gt.Equation()
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# Calculate the vertical derivative of the sedimentation flux
		# via the upwind method
		out_dfdz = gt.Equation(name='tmp_dfdz')
		out_dfdz[k] = (rho[k-1] * qr[k-1] * vt[k-1] -
					   rho[k  ] * qr[k  ] * vt[k  ]) / \
					  (tmp_h[k-1] - tmp_h[k])

		return out_dfdz


class _SecondOrderUpwind(SedimentationFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
	to implement a second-order accurate upwind method to discretize
	the vertical derivative of the sedimentation flux.

	Attributes
	----------
	nb : int
		Extent of the stencil in the upward vertical direction.
	"""
	def __init__(self):
		"""
		Note
		----
		To instantiate an object of this class, one should prefer
		the static method
		:meth:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux.factory`
		of :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 2

	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		# Instantiate temporary and output fields
		tmp_h    = gt.Equation()
		tmp_a    = gt.Equation()
		tmp_b    = gt.Equation()
		tmp_c    = gt.Equation()
		out_dfdz = gt.Equation(name='tmp_dfdz')

		# Interpolate the geometric height at the model main levels
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# Evaluate the space-dependent coefficients occurring in the
		# second-order, upwind finite difference approximation of the
		# vertical derivative of the flux
		tmp_a[k] = (2. * tmp_h[k] - tmp_h[k-1] - tmp_h[k-2]) / \
				   ((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k]))
		tmp_b[k] = (tmp_h[k-2] - tmp_h[k]) / \
				   ((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))
		tmp_c[k] = (tmp_h[k] - tmp_h[k-1]) / \
				   ((tmp_h[k-2] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))

		# Calculate the vertical derivative of the sedimentation flux
		# via the upwind method
		out_dfdz[k] = tmp_a[k] * rho[k  ] * qr[k  ] * vt[k  ] + \
					  tmp_b[k] * rho[k-1] * qr[k-1] * vt[k-1] + \
					  tmp_c[k] * rho[k-2] * qr[k-2] * vt[k-2]

		return out_dfdz


class Sedimentation(ImplicitTendencyComponent):
	"""
	This class inherits :class:`sympl.ImplicitTendencyComponent`
	to calculate the vertical derivative of the sedimentation flux.
	"""
	_d_physical_constants = {
		'density_of_liquid_water':
			DataArray(1e3, attrs={'units': 'kg m^-3'}),
	}

	def __init__(
		self, grid, sedimentation_flux_scheme, maximum_vertical_cfl=0.975,
		backend=gt.mode.NUMPY, physical_constants=None, **kwargs):
		"""
		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` object representing
			the underlying computational grid.
		sedimentation_flux_scheme : str
			The numerical sedimentation flux scheme. Please refer to
			:class:`~tasmania.physics.microphysics.SedimentationFlux`
			for the available options.
		maximum_vertical_cfl : `float`, optional
			Maximum allowed vertical CFL number. Defaults to 0.975.
		backend : `obj`, optional
			Backend for the underlying GT4Py stencils.
			Defaults to :obj:`gridtools.mode.NUMPY`.
        physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'density_of_liquid_water', in units compatible with [kg m^-3].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.physics.microphysics.Sedimentation._d_physical_constants`
			for the default values.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.DiagnosticComponent`.
		"""
		self._grid    = grid
		self._max_cfl = maximum_vertical_cfl
		self._backend = backend

		super().__init__(**kwargs)

		self._sflux = SedimentationFlux.factory(sedimentation_flux_scheme)

		self._physical_constants = get_physical_constants(
			self._d_physical_constants, physical_constants
		)

		self._stencil_sedimentation = None
		self._stencil_precipitation = None

	@property
	def input_properties(self):
		g = self._grid
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
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1 s^-1'}
		}

	@property
	def diagnostic_properties(self):
		dims2d = (self._grid.x.dims[0], self._grid.y.dims[0])

		return {
			'precipitation': {'dims': dims2d, 'units': 'mm hr^-1'}
		}

	def array_call(self, state, timestep):
		if self._stencil_precipitation is None:
			dtype = state['air_density'].dtype
			self._stencils_initialize(dtype)

		self._stencils_set_inputs(state, timestep)

		self._stencil_precipitation.compute()
		self._stencil_sedimentation.compute()

		tendencies = {
			'mass_fraction_of_precipitation_water_in_air': self._out_qr
		}
		diagnostics = {
			'precipitation': self._out_prec[:, :, 0]
		}

		return tendencies, diagnostics

	def _stencils_initialize(self, dtype):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		self._dt = gt.Global()
		self._maxcfl = gt.Global(self._max_cfl)
		self._rhow = gt.Global(self._physical_constants['density_of_liquid_water'])

		self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_h   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._in_qr  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_vt  = np.zeros((nx, ny, nz), dtype=dtype)

		self._out_qr   = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_prec = np.zeros((nx, ny, 1), dtype=dtype)

		self._stencil_sedimentation = gt.NGStencil(
			definitions_func= self._stencil_sedimentation_defs,
			inputs={
				'in_rho': self._in_rho, 'in_h': self._in_h,
				'in_qr': self._in_qr, 'in_vt': self._in_vt
			},
			global_inputs={'dt': self._dt, 'max_cfl': self._maxcfl},
			outputs={'out_qr': self._out_qr},
			domain=gt.domain.Rectangle((0, 0, self._sflux.nb), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

		self._stencil_precipitation = gt.NGStencil(
			definitions_func= self._stencil_precipitation_defs,
			inputs={
				'in_rho': self._in_rho[:, :, -1:], 'in_h': self._in_h[:, :, -2:],
				'in_qr': self._in_qr[:, :, -1:], 'in_vt': self._in_vt[:, :, -1:]
			},
			global_inputs={
				'dt': self._dt, 'max_cfl': self._maxcfl, 'rhow': self._rhow
			},
			outputs={'out_prec': self._out_prec},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, 0)),
			mode=self._backend
		)

	def _stencils_set_inputs(self, state, timestep):
		self._dt.value = timestep.total_seconds()

		self._in_rho[...] = state['air_density'][...]
		self._in_h[...]   = state['height_on_interface_levels'][...]
		self._in_qr[...]  = state['mass_fraction_of_precipitation_water_in_air'][...]
		self._in_vt[...]  = state['raindrop_fall_velocity'][...]

	def _stencil_sedimentation_defs(self, dt, max_cfl, in_rho, in_h, in_qr, in_vt):
		k = gt.Index(axis=2)

		tmp_dh = gt.Equation()
		tmp_vt = gt.Equation()

		out_qr = gt.Equation()

		tmp_dh[k] = in_h[k] - in_h[k+1]
		tmp_vt[k] = (in_vt[k] >  max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
				    (in_vt[k] <= max_cfl * tmp_dh[k] / dt) * in_vt[k]

		tmp_dfdz = self._sflux(k, in_rho, in_h, in_qr, tmp_vt)

		out_qr[k] = tmp_dfdz[k] / in_rho[k]

		return out_qr

	@staticmethod
	def _stencil_precipitation_defs(dt, max_cfl, rhow, in_rho, in_h, in_qr, in_vt):
		k = gt.Index(axis=2)

		tmp_dh = gt.Equation()
		tmp_vt = gt.Equation()

		out_prec = gt.Equation()

		tmp_dh[k] = in_h[k] - in_h[k+1]
		tmp_vt[k] = (in_vt[k] > max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
					(in_vt[k] > max_cfl * tmp_dh[k] / dt) * in_vt[k]

		out_prec[k] = in_rho[k] * in_qr[k] * tmp_vt[k] / (3.6e-6 * rhow)

		return out_prec
