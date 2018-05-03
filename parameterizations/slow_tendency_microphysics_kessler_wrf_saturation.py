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
import abc
import numpy as np

import gridtools as gt
from tasmania.namelist import cp, datatype, L, Rd, Rv
from tasmania.parameterizations.slow_tendencies import SlowTendencyMicrophysics
from tasmania.storages.grid_data import GridData
import tasmania.utils.utils as utils
import tasmania.utils.utils_meteo as utils_meteo

class SlowTendencyMicrophysicsKesslerWRFSaturation(SlowTendencyMicrophysics):
	"""
	This class inherits :class:`~tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics` 
	to implement the WRF version of the Kessler scheme.

	Note
	----
	The calculated tendencies do include the source terms deriving from the saturation adjustment.

	Attributes
	----------
	a : float
		Autoconversion threshold, in units of [:math:`g ~ kg^{-1}`].
	k1 : float
		Rate of autoconversion, in units of [:math:`s^{-1}`].
	k2 : float
		Rate of autoaccretion, in units of [:math:`s^{-1}`].
	"""
	def __init__(self, grid, rain_evaporation_on, backend, a = .5, k1 = .001, k2 = 2.2):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		rain_evaporation_on : bool
			:obj:`True` if the evaporation of raindrops should be taken into account, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.

		Keyword arguments
		-----------------
		a : float
			Autoconversion threshold, in units of [:math:`g ~ kg^{-1}`]. Default is :math:`0.5 ~ g ~ kg^{-1}`.
		k1 : float
			Rate of autoconversion, in units of [:math:`s^{-1}`]. Default is :math:`0.001 ~ s^{-1}`.
		k2 : float
			Rate of autoaccretion, in units of [:math:`s^{-1}`]. Default is :math:`2.2 ~ s^{-1}`.

		Note
		----
		To instantiate this class, one should prefer the static method 
		:meth:`~tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics.factory` of 
		:class:`~tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics`.
		"""
		super().__init__(grid, rain_evaporation_on, backend)
		self.a, self.k1, self.k2 = a, k1, k2

		# Constants for Tetens formula
		self._p0 = 610.78
		self._alpha = 17.27
		self._Tr = 273.15
		self._bw = 35.85
		
		# Shortcuts
		self._beta   = Rd / Rv
		self._beta_c = 1. - self._beta
		self._kappa  = L * self._alpha * self._beta * (self._Tr - self._bw) / cp 

		# Initialize pointers to the underlying GT4Py stencils
		# They will be properly re-directed the first time the entry point method is invoked
		self._stencil_tendency = None
		self._stencil_raindrop_fall_velocity = None

	def __call__(self, dt, state):
		"""
		Entry-point method computing slow-varying cloud microphysics tendencies.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the timestep.
		state : obj
			:class:`~tasmania.storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* exner_function (:math:`z`-staggered);
			* air_temperature (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).

		Returns
		-------
		tendencies : obj
			:class:`~tasmania.storages.grid_data.GridData` storing the following tendencies:

			* tendency_of_air_potential_temperature (unstaggered; only if rain evaporation is switched on);
			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered; only if rain evaporation is switched on);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).

		diagnostics : obj
			An empty :class:`~tasmania.storages.grid_data.GridData`, since no diagnostic is computed.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_tendency is None:
			self._stencil_tendency_initialize()

		# Update the local time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Extract the required model variables
		self._in_rho[:,:,:] = state['air_density'].values[:,:,:,0]
		self._in_p[:,:,:]   = state['air_pressure'].values[:,:,:,0]
		self._in_exn[:,:,:] = state['exner_function'].values[:,:,:,0]
		self._in_T[:,:,:]	= state['air_temperature'].values[:,:,:,0]
		self._in_qv[:,:,:]  = state['mass_fraction_of_water_vapor_in_air'].values[:,:,:,0]
		self._in_qc[:,:,:]  = state['mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0]
		self._in_qr[:,:,:]  = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Compute the saturation water vapor pressure via Tetens' formula
		self._in_ps[:,:,:] = self._p0 * np.exp(self._alpha * (self._in_T[:,:,:] - self._Tr) / \
											   (self._in_T[:,:,:] - self._bw))

		# Run the stencil
		self._stencil_tendency.compute()

		# Collect the tendencies
		time = utils.convert_datetime64_to_datetime(state['air_density'].coords['time'].values[0])
		tendencies = GridData(time, self._grid, 
						   	  tendency_of_mass_fraction_of_water_vapor_in_air         = self._out_qv_tnd,
							  tendency_of_mass_fraction_of_cloud_liquid_water_in_air  = self._out_qc_tnd,
							  tendency_of_mass_fraction_of_precipitation_water_in_air = self._out_qr_tnd,
							  tendency_of_air_potential_temperature                   = self._out_w)

		# Instantiate an empty GridData, acting as the output diagnostic
		diagnostics = GridData(time, self._grid)

		return tendencies, diagnostics

	def get_raindrop_fall_velocity(self, state):
		"""
		Get the raindrop fall velocity.

		Parameters
		----------
		state : obj
			:class:`~storages.grid_data.GridData` or one of its derived classes representing the current state.
			It should contain the following variables:

			* air_density (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered).
			
		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the raindrop fall velocity.
		"""
		# If this is the first time this method is invoked, initialize the auxiliary GT4Py stencil
		if self._stencil_raindrop_fall_velocity is None:
			self._stencil_raindrop_fall_velocity_initialize()
			
		# Extract the needed model variables
		self._in_rho[:,:,:] = state['air_density'].values[:,:,:,0]
		self._in_qr[:,:,:]  = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

		# Extract the surface density
		rho_s = self._in_rho[:,:,-1:]
		self._in_rho_s[:,:,:] = np.repeat(rho_s, self._grid.nz, axis = 2)

		# Call the stencil's compute function
		self._stencil_raindrop_fall_velocity.compute()

		return self._out_vt


	def _stencil_tendency_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of calculating the cloud microphysical tendencies.
		"""
		# Initialize the GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_p   = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_ps  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_exn = np.zeros((nx, ny, nz+1), dtype = datatype)
		self._in_T   = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qv  = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_qc  = np.zeros((nx, ny, nz), dtype = datatype)
		if self._stencil_raindrop_fall_velocity is None:
			self._in_rho = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_qr  = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy arrays which will serve as stencil outputs
		self._out_qv_tnd = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qc_tnd = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_qr_tnd = np.zeros((nx, ny, nz), dtype = datatype)
		self._out_w      = np.zeros((nx, ny, nz), dtype = datatype)

		# Set stencil's inputs and outputs
		_inputs  = {'in_rho': self._in_rho, 'in_p': self._in_p, 'in_ps': self._in_ps, 
				    'in_exn': self._in_exn, 'in_T': self._in_T,
				    'in_qv': self._in_qv, 'in_qc': self._in_qc, 'in_qr': self._in_qr}
		_outputs = {'out_qv_tnd': self._out_qv_tnd, 'out_qc_tnd': self._out_qc_tnd, 
					'out_qr_tnd': self._out_qr_tnd, 'out_w': self._out_w}

		# Initialize the stencil
		self._stencil_tendency = gt.NGStencil(
			definitions_func = self._stencil_tendency_defs,
			inputs = _inputs, 
			global_inputs = {'dt': self._dt},
			outputs = _outputs,
			domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
			mode = self._backend)

	def _stencil_tendency_defs(self, dt, in_rho, in_p, in_ps, in_exn, in_T, in_qv, in_qc, in_qr):
		"""
		GT4Py stencil calculating the cloud microphysical tendencies.

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_p : obj
			:class:`gridtools.Equation` representing the air pressure.
		in_ps : obj
			:class:`gridtools.Equation` representing the saturation vapor pressure.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_T : obj
			:class:`gridtools.Equation` representing the air temperature.
		in_qv : obj
			:class:`gridtools.Equation` representing the mass fraction of water vapor.
		in_qc : obj
			:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		out_qv_tnd : obj
			:class:`gridtools.Equation` representing the tendency of the mass fraction of water vapor.
		out_qc_tnd : obj
			:class:`gridtools.Equation` representing the tendency of the mass fraction of cloud liquid water.
		out_qr_tnd : obj
			:class:`gridtools.Equation` representing the tendency of the mass fraction of precipitation water.
		out_w : obj
			:class:`gridtools.Equation` representing the change over time in potential temperature.

		References
		----------
		Doms, G., et al. (2015). *A description of the nonhydrostatic regional COSMO-model. \
			Part II: Physical parameterization.* Retrieved from `http://www.cosmo-model.org`_. \
		Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
			*Compute Unified Device Architecture (CUDA)-based parallelization of WRF Kessler \
			cloud microphysics scheme*. Computer \& Geosciences, 52:292-299.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the temporary fields
		tmp_p	     = gt.Equation()
		tmp_p_mbar   = gt.Equation()
		tmp_rho_gcm3 = gt.Equation()
		tmp_qvs      = gt.Equation()
		tmp_Ar       = gt.Equation()
		tmp_Cr       = gt.Equation()
		tmp_C        = gt.Equation()
		tmp_Er       = gt.Equation()
		tmp_sat		 = gt.Equation()
		tmp_dlt		 = gt.Equation()

		# Instantiate the output fields
		out_qv_tnd = gt.Equation()
		out_qc_tnd = gt.Equation()
		out_qr_tnd = gt.Equation()
		out_w      = gt.Equation()

		# Interpolate the pressure at the model main levels
		tmp_p[i, j, k] = 0.5 * (in_p[i, j, k] + in_p[i, j, k+1])

		# Perform units conversion
		tmp_rho_gcm3[i, j, k] = 1.e3 * in_rho[i, j, k]
		tmp_p_mbar[i, j, k] = 1.e-2 * tmp_p[i, j, k]

		# Compute the saturation mixing ratio of water vapor
		tmp_qvs[i, j, k] = self._beta * in_ps[i, j, k] / (tmp_p[i, j, k] - self._beta_c * in_ps[i, j, k])

		# Compute the contribution of autoconversion to rain development
		tmp_Ar[i, j, k] = self.k1 * (in_qc[i, j, k] > self.a) * (in_qc[i, j, k] - self.a)

		# Compute the contribution of accretion to rain development
		tmp_Cr[i, j, k] = self.k2 * in_qc[i, j, k] * (in_qr[i, j, k] ** 0.875)

		if self._rain_evaporation_on:
			# Compute the contribution of evaporation to rain development
			tmp_C[i, j, k]  = 1.6 + 124.9 * ((tmp_rho_gcm3[i, j, k] * in_qr[i, j, k]) ** 0.2046)
			tmp_Er[i, j, k] = (1. - in_qv[i, j, k] / tmp_qvs[i, j, k]) * tmp_C[i, j, k] * \
							  ((tmp_rho_gcm3[i, j, k] * in_qr[i, j, k]) ** 0.525) / \
							  (tmp_rho_gcm3[i, j, k] * (5.4e5 + 2.55e6 / (tmp_p_mbar[i, j, k] * tmp_qvs[i, j, k])))

		# Compute the amount of latent heat released by the condensation of cloud liquid water
		tmp_sat[i, j, k] = (tmp_qvs[i, j, k] - in_qv[i, j, k]) / \
						   (1. + self._kappa * in_ps[i, j, k] / 
							((in_T[i, j, k] - self._bw) * (in_T[i, j, k] - self._bw) * 
							 (in_p[i, j, k] - self._beta * in_ps[i, j, k]) * 
							 (in_p[i, j, k] - self._beta * in_ps[i, j, k])))

		# Compute the source term representing the evaporation of cloud liquid water
		tmp_dlt[i, j, k] = ((tmp_sat[i, j, k] <= in_qc[i, j, k]) * tmp_sat[i, j, k] + \
						   	(tmp_sat[i, j, k] > in_qc[i, j, k]) * in_qc[i, j, k]) / (self.time_levels * dt)

		# Calculate the tendencies
		if not self._rain_evaporation_on:
			out_qv_tnd[i, j, k] = tmp_dlt[i, j, k]
			out_qc_tnd[i, j, k] = - (tmp_Ar[i, j, k] + tmp_Cr[i, j, k] + tmp_dlt[i, j, k])
			out_qr_tnd[i, j, k] = tmp_Ar[i, j, k] + tmp_Cr[i, j, k]
		else:
			out_qv_tnd[i, j, k] = tmp_dlt[i, j, k] + tmp_Er[i, j, k]
			out_qc_tnd[i, j, k] = - (tmp_Ar[i, j, k] + tmp_Cr[i, j, k] + tmp_dlt[i, j, k])
			out_qr_tnd[i, j, k] = tmp_Ar[i, j, k] + tmp_Cr[i, j, k] - tmp_Er[i, j, k]

		# Compute the change over time in potential temperature
		if not self._rain_evaporation_on:
			out_w[i, j, k] = - L / (.5 * (in_exn[i, j, k] + in_exn[i, j, k+1])) * tmp_dlt[i, j, k]
		else:
			out_w[i, j, k] = - L / (.5 * (in_exn[i, j, k] + in_exn[i, j, k+1])) * (tmp_dlt[i, j, k] + tmp_Er[i, j, k])

		return out_qv_tnd, out_qc_tnd, out_qr_tnd, out_w 

	def _stencil_raindrop_fall_velocity_initialize(self):
		"""
		Initialize the GT4Py stencil calculating the raindrop velocity.
		"""
		# Allocate the Numpy arrays which will serve as stencil inputs
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		self._in_rho_s = np.zeros((nx, ny, nz), dtype = datatype)
		if self._stencil_tendency is None:
			self._in_rho = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_qr  = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate the Numpy array which will serve as stencil output
		self._out_vt = np.zeros((nx, ny, nz), dtype = datatype)

		# Initialize the stencil
		self._stencil_raindrop_fall_velocity = gt.NGStencil(
				definitions_func = self._stencil_raindrop_fall_velocity_defs,
				inputs = {'in_rho': self._in_rho, 'in_rho_s': self._in_rho_s, 'in_qr': self._in_qr}, 
				outputs = {'out_vt': self._out_vt},
				domain = gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
				mode = self._backend)

	def _stencil_raindrop_fall_velocity_defs(self, in_rho, in_rho_s, in_qr):
		"""
		GT4Py stencil calculating the raindrop velocity. 

		Parameters
		----------
		in_rho : obj
			:class:`gridtools.Equation` representing the density.
		in_rho_s : obj
			:class:`gridtools.Equation` representing the surface density.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the raindrop fall velocity.
		"""
		# Declare the indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate the output field
		out_vt = gt.Equation()

		# Perform computations
		out_vt[i, j, k] = 36.34 * (1.e-3 * in_rho[i, j, k] * (in_qr[i, j, k] > 0.) * in_qr[i, j, k]) ** 0.1346 * \
						  ((in_rho_s[i, j, k] / in_rho[i, j, k]) ** 0.5)

		return out_vt
