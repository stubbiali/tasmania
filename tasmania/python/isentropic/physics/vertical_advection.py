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
	NGIsentropicMinimalVerticalFlux
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
		self, domain, flux_scheme='upwind', tracers=None,
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
		tracers : `ordered dict`, optional
			(Ordered) dictionary whose keys are strings denoting the tracers
			included in the model, and whose values are	dictionaries specifying
			fundamental properties ('units', 'stencil_symbol') for those tracers.
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
		self._tracers = {} if tracers is None else tracers
		self._stgz    = tendency_of_air_potential_temperature_on_interface_levels

		# call parent's constructor
		super().__init__(domain, 'numerical', **kwargs)

		# instantiate the object calculating the flux
		self._vflux = NGIsentropicMinimalVerticalFlux.factory(
			flux_scheme, self.grid, tracers
		)

		# initialize the underlying GT4Py stencil
		# remark: thanks to _vflux, now each dictionary in _tracers has the
		# 'stencil_symbol' key
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

		for tracer, props in self._tracers.items():
			return_dict[tracer] = {'dims': dims, 'units': props['units']}

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
		for tracer, props in self._tracers.items():
			return_dict[tracer] = {'dims': dims, 'units': props['units'] + ' s^-1'}

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
		tendencies.update({tracer: self._out_q[tracer] for tracer in self._tracers})

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
		self._in_q  = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

		# allocate arrays serving as stencil's outputs
		self._out_s  = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
		self._out_q  = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

		# set stencil's inputs
		inputs = {
			'in_w': self._in_w, 'in_s': self._in_s,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		inputs.update({
			'in_' + props['stencil_symbol']: self._in_q[tracer]
			for tracer, props in self._tracers.items()
		})

		# set stencil's outputs
		outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv,
		}
		outputs.update({
			'out_' + props['stencil_symbol']: self._out_q[tracer]
			for tracer, props in self._tracers.items()
		})

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

		for tracer in self._tracers:
			self._in_q[tracer][...] = state[tracer][...]

	def _stencil_defs(self, in_w, in_s, in_su, in_sv, **tracer_kwargs):
		# shortcuts
		dz = self._grid.dz.to_units('K').values.item()
		ts = {
			tracer: props['stencil_symbol']
			for tracer, props in self._tracers.items()
		}

		# vertical index
		k = gt.Index(axis=2)

		# temporary fields
		in_sq = {
			's' + ts[tracer]: gt.Equation(name='in_s' + ts[tracer])
			for tracer in ts
		}

		# output fields
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		out_q  = {
			tracer: gt.Equation(name='out_' + ts[tracer])
			for tracer in ts
		}

		# retrieve tracers
		in_q = {
			tracer: tracer_kwargs['in_' + ts[tracer]]
			for tracer in ts
		}

		# vertical velocity at the interface levels
		if self._stgz:
			w = in_w
		else:
			w = gt.Equation()
			w[k] = 0.5 * (in_w[k] + in_w[k-1])

		# vertical fluxes
		for tracer in ts:
			in_sq['s' + ts[tracer]][k] = in_s[k] * in_q[tracer][k]
		fluxes = self._vflux(k, w, in_s, in_su, in_sv, **in_sq)

		# vertical advection
		out_s[k]  = (fluxes[0][k+1] - fluxes[0][k]) / dz
		out_su[k] = (fluxes[1][k+1] - fluxes[1][k]) / dz
		out_sv[k] = (fluxes[2][k+1] - fluxes[2][k]) / dz
		for idx, tracer in enumerate(ts.keys()):
			out_q[tracer][k] = (fluxes[3 + idx][k+1] - fluxes[3 + idx][k]) / \
							   (in_s[k] * dz)

		return_list = [out_s, out_su, out_sv] + [out_q[tracer] for tracer in ts]

		return return_list

	def _set_lower_layers(self):
		dz = self._grid.dz.to_units('K').values.item()
		nb, order = self._vflux.extent, self._vflux.order

		w = self._in_w if not self._stgz \
			else 0.5 * (self._in_w[:, :, -nb-2:] + self._in_w[:, :, -nb-3:-1])

		s  = self._in_s
		su = self._in_su
		sv = self._in_sv
		q  = self._in_q

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

			for tracer in self._tracers:
				self._out_q[tracer][:, :, -nb:] = (
					w[:, :, -nb-1:-1] * (s[:, :, -nb-1:-1] * q[tracer][:, :, -nb-1:-1]) -
					w[:, :, -nb:    ] * (s[:, :, -nb:    ] * q[tracer][:, :, -nb:    ])
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

			for tracer in self._tracers:
				self._out_q[tracer][:, :, -nb:] = 0.5 * (
					- 3.0 * w[:, :, -nb:    ] * (s[:, :, -nb:    ] * q[tracer][:, :, -nb:    ])
					+ 4.0 * w[:, :, -nb-1:-1] * (s[:, :, -nb-1:-1] * q[tracer][:, :, -nb-1:-1])
					- 1.0 * w[:, :, -nb-2:-2] * (s[:, :, -nb-2:-2] * q[tracer][:, :, -nb-2:-2])
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
