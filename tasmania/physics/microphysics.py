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
from sympl import DataArray, \
				  DiagnosticComponent as Diagnostic, \
				  TendencyComponent as Tendency

import gridtools as gt
from tasmania.utils.data_utils import get_physical_constants


class Kessler(Tendency):
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

	def __init__(self, grid, pressure_on_interface_levels=True,
				 rain_evaporation_on=True, autoconversion_threshold=_d_a,
				 rate_of_autoconversion=_d_k1, rate_of_collection=_d_k2,
				 backend=gt.mode.NUMPY, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes.
		pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		rain_evaporation_on : `bool`, optional
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
		"""
		# Keep track of input arguments
		self._grid = grid
		self._pressure_on_interface_levels = pressure_on_interface_levels
		self._rain_evaporation_on = rain_evaporation_on
		self._a = autoconversion_threshold.to_units('g g^-1').values.item()
		self._k1 = rate_of_autoconversion.to_units('s^-1').values.item()
		self._k2 = rate_of_collection.to_units('s^-1').values.item()
		self._backend = backend

		# Call parent's constructor
		super().__init__()

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
		dims_on_interface_levels = (grid.x.dims[0],
									grid.y.dims[0],
									grid.z_on_interface_levels.dims[0])

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

		if self._pressure_on_interface_levels:
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

		if self._rain_evaporation_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['air_potential_temperature'] = \
				{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
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
		if self._pressure_on_interface_levels:
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
		if self._rain_evaporation_on:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv_tnd
			tendencies['air_potential_temperature'] = self._out_theta_tnd

		# Instantiate an empty dictionary, serving as the output diagnostics dictionary
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
		if self._pressure_on_interface_levels:
			self._in_p   = np.zeros((nx, ny, nz+1), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=dtype)
		else:
			self._in_p   = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_exn = np.zeros((nx, ny, nz), dtype=dtype)

		# Allocate the numpy arrays which will serve as stencil outputs
		self._out_qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)
		if self._rain_evaporation_on:
			self._out_qv_tnd    = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_theta_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# Set stencil's inputs and outputs
		_inputs  = {'in_rho': self._in_rho, 'in_p': self._in_p,
					'in_ps': self._in_ps, 'in_exn': self._in_exn,
				    'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr}
		_outputs = {'out_qc_tnd': self._out_qc_tnd, 'out_qr_tnd': self._out_qr_tnd}
		if self._rain_evaporation_on:
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
		if self._pressure_on_interface_levels:
			tmp_p 	 = gt.Equation()
			tmp_exn	 = gt.Equation()
		if self._rain_evaporation_on:
			tmp_c    = gt.Equation()
			tmp_er   = gt.Equation()

		# Instantiate the output fields
		out_qc_tnd = gt.Equation()
		out_qr_tnd = gt.Equation()
		if self._rain_evaporation_on:
			out_qv_tnd    = gt.Equation()
			out_theta_tnd = gt.Equation()

		# Interpolate the pressure and the Exner function at the vertical main levels
		if self._pressure_on_interface_levels:
			tmp_p[k]   = 0.5 * (in_p[k] + in_p[k+1])
			tmp_exn[k] = 0.5 * (in_exn[k] + in_exn[k+1])

		# Set pointers to equations representing pressure and Exner function
		# at the main levels
		p   = tmp_p if self._pressure_on_interface_levels else in_p
		exn = tmp_exn if self._pressure_on_interface_levels else in_exn

		# Perform units conversion
		tmp_rho_gcm3[k] = 1.e3 * in_rho[k]
		tmp_p_mbar[k]   = 1.e-2 * p[k]

		# Compute the saturation mixing ratio of water vapor
		tmp_qvs[k] = self._beta * in_ps[k] / (p[k] - self._beta_c * in_ps[k])

		# Compute the contribution of autoconversion to rain development
		tmp_ar[k] = self._k1 * (in_qc[k] > self._a) * (in_qc[k] - self._a)

		# Compute the contribution of accretion to rain development
		tmp_cr[k] = self._k2 * in_qc[k] * (in_qr[k] ** 0.875)

		if self._rain_evaporation_on:
			# Compute the contribution of evaporation to rain development
			tmp_c[k]  = 1.6 + 124.9 * ((tmp_rho_gcm3[k] * in_qr[k]) ** 0.2046)
			tmp_er[k] = (1. - in_qv[k] / tmp_qvs[k]) * tmp_c[k] * \
						((tmp_rho_gcm3[k] * in_qr[k]) ** 0.525) / \
						(tmp_rho_gcm3[k] * (5.4e5 + 2.55e6 / (tmp_p_mbar[k] * tmp_qvs[k])))

		# Calculate the tendencies
		if not self._rain_evaporation_on:
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k]
		else:
			out_qv_tnd[k] = tmp_er[k]
			out_qc_tnd[k] = - (tmp_ar[k] + tmp_cr[k])
			out_qr_tnd[k] = tmp_ar[k] + tmp_cr[k] - tmp_er[k]

		# Compute the change over time in potential temperature
		if self._rain_evaporation_on:
			lhvw = self._physical_constants['latent_heat_of_vaporization_of_water']
			out_theta_tnd[k] = - lhvw / exn[k] * tmp_er[k]

		if not self._rain_evaporation_on:
			return out_qc_tnd, out_qr_tnd
		else:
			return out_qc_tnd, out_qr_tnd, out_qv_tnd, out_theta_tnd


class SaturationAdjustmentKessler(Diagnostic):
	"""
	This class inherits :class:`sympl.Diagnostic` to implement the saturation
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

	def __init__(self, grid, pressure_on_interface_levels=True,
				 backend=gt.mode.NUMPY, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			 or one of its derived classes.
		pressure_on_interface_levels : `bool`, optional
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
		"""
		# Keep track of input arguments
		self._grid = grid
		self._pressure_on_interface_levels = pressure_on_interface_levels
		self._backend = backend

		# Call parent's constructor
		super().__init__()

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
		dims_on_interface_levels = (grid.x.dims[0],
									grid.y.dims[0],
									grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_temperature':
				{'dims': dims, 'units': 'K'},
			'mass_fraction_of_water_vapor_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'mass_fraction_of_cloud_liquid_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
		}

		if self._pressure_on_interface_levels:
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
		if self._pressure_on_interface_levels:
			self._in_p[...]   = state['air_pressure_on_interface_levels'][...]
		else:
			self._in_p[...]   = state['air_pressure'][...]

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
		if self._pressure_on_interface_levels:
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
		if self._pressure_on_interface_levels:
			tmp_p = gt.Equation()

		# Instantiate the output fields
		out_qv = gt.Equation()
		out_qc = gt.Equation()

		# Interpolate the pressure at the vertical main levels
		if self._pressure_on_interface_levels:
			tmp_p[k] = 0.5 * (in_p[k] + in_p[k+1])

		# Set the pointer to the equation representing the pressure
		p = tmp_p if self._pressure_on_interface_levels else in_p

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


class RaindropFallVelocity(Diagnostic):
	"""
	This class inherits :class:`sympl.Diagnostic` to calculate
	the raindrop fall velocity.
	"""
	def __init__(self, grid, backend=gt.mode.NUMPY):
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
		"""
		# Keep track of input arguments
		self._grid = grid
		self._backend = backend

		# Call parent's constructor
		super().__init__()

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
			inputs			 = {'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s,
								'in_qr': self._in_qr},
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

