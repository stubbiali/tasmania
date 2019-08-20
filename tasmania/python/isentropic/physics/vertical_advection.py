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
	IsentropicVerticalAdvection
	PrescribedSurfaceHeating
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.isentropic.dynamics.vertical_fluxes import \
	IsentropicMinimalVerticalFlux
from tasmania.python.utils.data_utils import get_physical_constants

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicVerticalAdvection(TendencyComponent):
	"""
	This class inherits :class:`tasmania.TendencyComponent` to calculate
	the vertical derivative of the conservative vertical advection flux
	in isentropic coordinates for any prognostic variable included in
	the isentropic model. The class is always instantiated over the
	numerical grid of the underlying domain.
	"""
	def __init__(
		self, domain, flux_scheme='upwind', moist=False,
		tendency_of_air_potential_temperature_on_interface_levels=False,
		backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		flux_scheme : `str`, optional
			The numerical flux scheme to implement. Defaults to 'upwind'.
			See :class:`~tasmania.IsentropicMinimalVerticalFlux` for all
			available options.
		moist : `bool`, optional
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
			:obj:`True` if the input tendency of air potential temperature
			is defined at the interface levels, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.TendencyComponent`.
		"""
		# keep track of input arguments
		self._moist = moist
		self._stgz  = tendency_of_air_potential_temperature_on_interface_levels

		# call parent's constructor
		super().__init__(domain, 'numerical', **kwargs)

		# instantiate the object calculating the flux
		self._vflux = IsentropicMinimalVerticalFlux.factory(flux_scheme, self.grid, moist)

		# initialize the underlying GT4Py stencil
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		grid = self.grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		if self._stgz:
			dims_stgz = (
				grid.x.dims[0], grid.y.dims[0], grid.z_on_interface_levels.dims[0]
			)
			return_dict['tendency_of_air_potential_temperature_on_interface_levels'] = \
				{'dims': dims_stgz, 'units': 'K s^-1'}
		else:
			return_dict['tendency_of_air_potential_temperature'] = \
				{'dims': dims, 'units': 'K s^-1'}

		if self._moist:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		grid = self.grid
		dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}
		if self._moist:
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

	def array_call(self, state):
		# set the stencil's inputs
		self._stencil_set_inputs(state)

		# run the stencil
		self._stencil.compute()

		# set lower layers
		self._set_lower_layers()

		# collect the output arrays in a dictionary
		tendencies = {
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist:
			tendencies['mass_fraction_of_water_vapor_in_air'] = self._out_qv
			tendencies['mass_fraction_of_cloud_liquid_water_in_air'] = self._out_qc
			tendencies['mass_fraction_of_precipitation_water_in_air'] = self._out_qr

		return tendencies, {}

	def _stencil_initialize(self, backend, dtype):
		# shortcuts
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		mk = nz + 1 if self._stgz else nz
		nb = self._vflux.extent

		# allocate arrays serving as stencil's inputs
		self._in_w  = np.zeros((nx, ny, mk), dtype=dtype)
		self._in_s  = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate arrays serving as stencil's outputs
		self._out_s  = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)

		# set stencil's inputs
		inputs = {
			'in_w': self._in_w, 'in_s': self._in_s,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		if self._moist:
			inputs['in_qv'] = self._in_qv
			inputs['in_qc'] = self._in_qc
			inputs['in_qr'] = self._in_qr

		# set stencil's outputs
		outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv,
		}
		if self._moist:
			outputs['out_qv'] = self._out_qv
			outputs['out_qc'] = self._out_qc
			outputs['out_qr'] = self._out_qr

		# instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=inputs,
			outputs=outputs,
			domain=gt.domain.Rectangle((0, 0, nb), (nx-1, ny-1, nz-nb-1)),
			mode=backend
		)

	def _stencil_set_inputs(self, state):
		if self._stgz:
			self._in_w[...] = \
				state['tendency_of_air_potential_temperature_on_interface_levels'][...]
		else:
			self._in_w[...] = state['tendency_of_air_potential_temperature'][...]

		self._in_s[...]  = state['air_isentropic_density'][...]
		self._in_su[...] = state['x_momentum_isentropic'][...]
		self._in_sv[...] = state['y_momentum_isentropic'][...]

		if self._moist:
			self._in_qv[...] = state['mass_fraction_of_water_vapor_in_air'][...]
			self._in_qc[...] = state['mass_fraction_of_cloud_liquid_water_in_air'][...]
			self._in_qr[...] = state['mass_fraction_of_precipitation_water_in_air'][...]

	def _stencil_defs(
		self, in_w, in_s, in_su, in_sv, in_qv=None, in_qc=None, in_qr=None
	):
		# shortcuts
		dz = self._grid.dz.to_units('K').values.item()

		# vertical index
		k = gt.Index(axis=2)

		# output fields
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			in_sqv = gt.Equation()
			in_sqc = gt.Equation()
			in_sqr = gt.Equation()
			out_qv = gt.Equation()
			out_qc = gt.Equation()
			out_qr = gt.Equation()

		# vertical velocity at the interface levels
		if self._stgz:
			w = in_w
		else:
			w = gt.Equation()
			w[k] = 0.5 * (in_w[k] + in_w[k-1])

		# vertical fluxes
		if not self._moist:
			flux_s, flux_su, flux_sv = self._vflux(k, w, in_s, in_su, in_sv)
		else:
			in_sqv[k] = in_s[k] * in_qv[k]
			in_sqc[k] = in_s[k] * in_qc[k]
			in_sqr[k] = in_s[k] * in_qr[k]
			flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = \
				self._vflux(k, w, in_s, in_su, in_sv, in_sqv, in_sqc, in_sqr)

		# vertical advection
		out_s[k]  = (flux_s[k+1]  - flux_s[k] ) / dz
		out_su[k] = (flux_su[k+1] - flux_su[k]) / dz
		out_sv[k] = (flux_sv[k+1] - flux_sv[k]) / dz
		if self._moist:
			out_qv[k] = (flux_sqv[k+1] - flux_sqv[k]) / (in_s[k] * dz)
			out_qc[k] = (flux_sqc[k+1] - flux_sqc[k]) / (in_s[k] * dz)
			out_qr[k] = (flux_sqr[k+1] - flux_sqr[k]) / (in_s[k] * dz)

		if not self._moist:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_qv, out_qc, out_qr

	def _set_lower_layers(self):
		dz = self._grid.dz.to_units('K').values.item()
		nb, order = self._vflux.extent, self._vflux.order

		w = self._in_w if not self._stgz \
			else 0.5 * (self._in_w[:, :, -nb-2:] + self._in_w[:, :, -nb-3:-1])

		s  = self._in_s
		su = self._in_su
		sv = self._in_sv
		if self._moist:
			qv = self._in_qv
			qc = self._in_qc
			qr = self._in_qr

		if order == 1:
			self._out_s[:, :, -nb:] = (
				w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] -
				w[:, :, -nb:    ] * s[:, :, -nb:    ]
			) / dz
			self._out_su[:, :, -nb:] = (
				w[:, :, -nb-1:-1] * su[:, :, -nb-1:-1] -
				w[:, :, -nb:    ] * su[:, :, -nb:    ]
			) / dz
			self._out_sv[:, :, -nb:] = (
				w[:, :, -nb-1:-1] * sv[:, :, -nb-1:-1] -
				w[:, :, -nb:    ] * sv[:, :, -nb:    ]
			) / dz

			if self._moist:
				self._out_qv[:, :, -nb:] = (
					w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qv[:, :, -nb-1:-1] -
					w[:, :, -nb:    ] * s[:, :, -nb:    ] * qv[:, :, -nb:    ]
				) / (dz * s[:, :, -nb:])
				self._out_qc[:, :, -nb:] = (
					w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qc[:, :, -nb-1:-1] -
					w[:, :, -nb:    ] * s[:, :, -nb:    ] * qc[:, :, -nb:    ]
				) / (dz * s[:, :, -nb:])
				self._out_qr[:, :, -nb:] = (
					w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qr[:, :, -nb-1:-1] -
					w[:, :, -nb:    ] * s[:, :, -nb:    ] * qr[:, :, -nb:    ]
				) / (dz * s[:, :, -nb:])
		else:
			self._out_s[:, :, -nb:] = 0.5 * (
				- 3.0 * w[:, :, -nb:    ] * s[:, :, -nb:    ]
				+ 4.0 * w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1]
				- 1.0 * w[:, :, -nb-2:-2] * s[:, :, -nb-2:-2]
			) / dz
			self._out_su[:, :, -nb:] = 0.5 * (
				- 3.0 * w[:, :, -nb:    ] * su[:, :, -nb:    ]
				+ 4.0 * w[:, :, -nb-1:-1] * su[:, :, -nb-1:-1]
				- 1.0 * w[:, :, -nb-2:-2] * su[:, :, -nb-2:-2]
			) / dz
			self._out_sv[:, :, -nb:] = 0.5 * (
				- 3.0 * w[:, :, -nb:    ] * sv[:, :, -nb:    ]
				+ 4.0 * w[:, :, -nb-1:-1] * sv[:, :, -nb-1:-1]
				- 1.0 * w[:, :, -nb-2:-2] * sv[:, :, -nb-2:-2]
			) / dz

			if self._moist:
				self._out_qv[:, :, -nb:] = 0.5 * (
					- 3.0 * w[:, :, -nb:    ] * s[:, :, -nb:    ] * qv[:, :, -nb:    ]
					+ 4.0 * w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qv[:, :, -nb-1:-1]
					- 1.0 * w[:, :, -nb-2:-2] * s[:, :, -nb-2:-2] * qv[:, :, -nb-2:-2]
				) / (dz * s[:, :, -nb:])
				self._out_qc[:, :, -nb:] = 0.5 * (
					- 3.0 * w[:, :, -nb:    ] * s[:, :, -nb:    ] * qc[:, :, -nb:    ]
					+ 4.0 * w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qc[:, :, -nb-1:-1]
					- 1.0 * w[:, :, -nb-2:-2] * s[:, :, -nb-2:-2] * qc[:, :, -nb-2:-2]
				) / (dz * s[:, :, -nb:])
				self._out_qr[:, :, -nb:] = 0.5 * (
					- 3.0 * w[:, :, -nb:    ] * s[:, :, -nb:    ] * qr[:, :, -nb:    ]
					+ 4.0 * w[:, :, -nb-1:-1] * s[:, :, -nb-1:-1] * qr[:, :, -nb-1:-1]
					- 1.0 * w[:, :, -nb-2:-2] * s[:, :, -nb-2:-2] * qr[:, :, -nb-2:-2]
				) / (dz * s[:, :, -nb:])


class PrescribedSurfaceHeating(TendencyComponent):
	"""
	Calculate the variation in air potential temperature as prescribed
	in the reference paper, namely

    	.. math::
        	\dot{\theta} =
        	\Biggl \lbrace
        	{
        		\\frac{\theta \, R_d \, \alpha(t)}{p \, C_p}
            	\exp[\left( - \alpha(t) \left( z - h_s \\right) \\right]}
            	\left[ F_0^{sw}(t) \sin{\left( \omega^{sw} (t - t_0) \\right)}
            	+ F_0^{fw}(t) \sin{\left( \omega^{fw} (t - t_0) \\right)} \\right]
            	\text{if} {r = \sqrt{x^2 + y^2} < R}
            	\atop
            	0 \text{otherwise}
            } .

	The class is always instantiated over the numerical grid of the
	underlying domain.

	References
	----------
	Reisner, J. M., and P. K. Smolarkiewicz. (1994). \
		Thermally forced low Froude number flow past three-dimensional obstacles. \
		*Journal of Atmospheric Sciences*, *51*(1):117-133.
	"""
	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'gas_constant_of_dry_air':
			DataArray(287.0, attrs={'units': 'J K^-1 kg^-1'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(
		self, domain,
		tendency_of_air_potential_temperature_in_diagnostics=False,
		tendency_of_air_potential_temperature_on_interface_levels=False,
		air_pressure_on_interface_levels=True,
		amplitude_at_day_sw=None, amplitude_at_day_fw=None, 
		amplitude_at_night_sw=None, amplitude_at_night_fw=None,
		frequency_sw=None, frequency_fw=None,
		attenuation_coefficient_at_day=None,
		attenuation_coefficient_at_night=None,
		characteristic_length=None, starting_time=None,
		backend=gt.mode.NUMPY, physical_constants=None, **kwargs):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			:obj:`True` to place the calculated tendency of air
			potential temperature in the ``diagnostics`` output
			dictionary, :obj:`False` to regularly place it in the
			`tendencies` dictionary. Defaults to :obj:`False`.
		tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the tendency
			of air potential temperature should be calculated at the
			interface (resp., main) vertical levels. Defaults to :obj:`False`.
		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input
			air potential pressure is defined at the interface
			(resp., main) vertical levels. Defaults to :obj:`True`.
		amplitude_at_day_sw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at day,
			in units compatible with [W m^-2].
		amplitude_at_day_fw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at day,
			in units compatible with [W m^-2].
		amplitude_at_night_sw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at night,
			in units compatible with [W m^-2].
		amplitude_at_night_fw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at night,
			in units compatible with [W m^-2].
		frequency_sw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`\omega^{sw}`,
			in units compatible with [s^-1].
		frequency_fw : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`\omega^{fw}`,
			in units compatible with [s^-1].
		attenuation_coefficient_at_day : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`\alpha` at day,
			in units compatible with [m^-1].
		attenuation_coefficient_at_night : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`\alpha` at night,
			in units compatible with [m^-1].
		characteristic_length : `sympl.DataArray`, optional
			1-item :class:`~sympl.DataArray` representing :math:`R`,
			in units compatible with [m].
		starting_time : `datetime`, optional
			The time :math:`t_0` when surface heating/cooling is triggered.
		backend : `obj`, optional
			TODO
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`~sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`sympl.TendencyComponent`.
		"""
		self._tid = tendency_of_air_potential_temperature_in_diagnostics
		self._apil = air_pressure_on_interface_levels
		self._aptil = tendency_of_air_potential_temperature_on_interface_levels \
			and air_pressure_on_interface_levels
		self._backend = backend

		super().__init__(domain, 'numerical', **kwargs)

		self._f0d_sw = amplitude_at_day_sw.to_units('W m^-2').values.item() \
			if amplitude_at_day_sw is not None else 800.0
		self._f0d_fw = amplitude_at_day_fw.to_units('W m^-2').values.item() \
			if amplitude_at_day_fw is not None else 400.0
		self._f0n_sw = amplitude_at_night_sw.to_units('W m^-2').values.item() \
			if amplitude_at_night_sw is not None else -75.0
		self._f0n_fw = amplitude_at_night_fw.to_units('W m^-2').values.item() \
			if amplitude_at_night_fw is not None else -37.5
		self._w_sw = frequency_sw.to_units('hr^-1').values.item() \
			if frequency_sw is not None else np.pi/12.0
		self._w_fw = frequency_fw.to_units('hr^-1').values.item() \
			if frequency_fw is not None else np.pi
		self._ad = attenuation_coefficient_at_day.to_units('m^-1').values.item() \
			if attenuation_coefficient_at_day is not None else 1.0/600.0
		self._an = attenuation_coefficient_at_night.to_units('m^-1').values.item() \
			if attenuation_coefficient_at_night is not None else 1.0/75.0
		self._cl = characteristic_length.to_units('m').values.item() \
			if characteristic_length is not None else 25000.0
		self._t0 = starting_time

		pcs = get_physical_constants(self._d_physical_constants, physical_constants)
		self._rd = pcs['gas_constant_of_dry_air']
		self._cp = pcs['specific_heat_of_dry_air_at_constant_pressure']

	@property
	def input_properties(self):
		g = self.grid
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
		g = self.grid

		return_dict = {}

		if not self._tid:
			if self._aptil:
				dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
				return_dict['air_potential_temperature_on_interface_levels'] = \
					{'dims': dims, 'units': 'K s^-1'}
			else:
				dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
				return_dict['air_potential_temperature'] = \
					{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid

		return_dict = {}

		if self._tid:
			if self._aptil:
				dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
				return_dict['tendency_of_air_potential_temperature_on_interface_levels'] = \
					{'dims': dims, 'units': 'K s^-1'}
			else:
				dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
				return_dict['tendency_of_air_potential_temperature'] = \
					{'dims': dims, 'units': 'K s^-1'}

		return return_dict

	def array_call(self, state):
		g = self.grid
		mi, mj = g.nx, g.ny
		mk = g.nz + 1 if self._aptil else g.nz

		t  = state['time']
		dt = (t - self._t0).total_seconds() / 3600.0 if self._t0 is not None else t.hour

		if dt <= 0.0:
			out = np.zeros((mi, mj, mk), dtype=state['height_on_interface_levels'].dtype)
		else:
			x = g.x.to_units('m').values[:, np.newaxis, np.newaxis]
			y = g.y.to_units('m').values[np.newaxis, :, np.newaxis]
			theta1d = g.z_on_interface_levels.to_units('K').values if self._aptil \
				else g.z.to_units('K').values
			theta = theta1d[np.newaxis, np.newaxis, :]

			pv = state['air_pressure_on_interface_levels'] if self._apil \
				else state['air_pressure']
			p  = pv if pv.shape[2] == mk else 0.5 * (pv[:, :, 1:] + pv[:, :, :-1])
			zv = state['height_on_interface_levels']
			z  = zv if self._aptil else 0.5 * (zv[:, :, 1:] + zv[:, :, :-1])
			h  = zv[:, :, -1:]

			w_sw  = self._w_sw
			w_fw  = self._w_fw
			cl    = self._cl

			t_in_seconds = t.hour*60*60 + t.minute*60 + t.second
			t_sw  = (2*np.pi / w_sw) * 60 * 60
			day   = int(t_in_seconds / t_sw) % 2 == 0
			f0_sw = self._f0d_sw if day else self._f0n_sw
			f0_fw = self._f0d_fw if day else self._f0n_fw	
			a  	  = self._ad if day else self._an

			out = theta * self._rd * a / (p * self._cp) * np.exp(- a * (z - h)) * \
				(f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt)) * \
				(x**2 + y**2 < cl**2)

		tendencies = {}
		if not self._tid:
			if self._aptil:
				tendencies['air_potential_temperature_on_interface_levels'] = out
			else:
				tendencies['air_potential_temperature'] = out

		diagnostics = {}
		if self._tid:
			if self._aptil:
				diagnostics['tendency_of_air_potential_temperature_on_interface_levels'] = out
			else:
				diagnostics['tendency_of_air_potential_temperature'] = out

		return tendencies, diagnostics
