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
from datetime import timedelta
from hypothesis import \
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest
from sympl import DataArray

import gridtools as gt
from tasmania.python.physics.microphysics import \
	Kessler, SaturationAdjustmentKessler, RaindropFallVelocity, \
	SedimentationFlux, _FirstOrderUpwind, _SecondOrderUpwind, Sedimentation, \
	Precipitation
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.meteo_utils import \
	goff_gratch_formula, tetens_formula

try:
	from .conf import backend as conf_backend
	from .utils import compare_arrays, compare_dataarrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f
except ModuleNotFoundError:
	from conf import backend as conf_backend
	from utils import compare_arrays, compare_dataarrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


def kessler_validation(
	rho, p, t, exn, qv, qc, qr, a, k1, k2, swvf, beta, lhvw, rain_evaporation
):
	p = p if p.shape[2] == rho.shape[2] else 0.5*(p[:, :, :-1] + p[:, :, 1:])
	exn = exn if exn.shape[2] == rho.shape[2] else 0.5*(exn[:, :, :-1] + exn[:, :, 1:])

	p_mbar = 0.01 * p
	rho_gcm3 = 0.001 * rho

	ar = k1 * (qc - a) * (qc > a)
	cr = k2 * qc * qr**0.875

	tnd_qc = - ar - cr
	tnd_qr = ar + cr

	if rain_evaporation:
		ps = swvf(t)
		qvs = beta * ps / (p - ps)
		c = 1.6 + 124.9 * (rho_gcm3 * qr)**0.2046
		er = (1.0 - qv / qvs) * c * (rho_gcm3 * qr)**0.525 / (
			rho_gcm3 * (5.4e5 + 2.55e6 / (p_mbar * qvs))
		)
		tnd_qv = er
		tnd_qr -= er
		tnd_theta = - lhvw * er / exn
	else:
		tnd_qv = 0.0
		tnd_theta = 0.0

	return tnd_qv, tnd_qc, tnd_qr, tnd_theta


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_kessler(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

	apoif = data.draw(hyp_st.booleans(), label="apoif")
	toaptid = data.draw(hyp_st.booleans(), label="toaptid")
	re = data.draw(hyp_st.booleans(), label="re")
	swvf_type = data.draw(st_one_of(('tetens', 'goff_gratch')), label="swvf_type")

	a = data.draw(hyp_st.floats(min_value=0, max_value=10), label="a")
	k1 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k1")
	k2 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k2")

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	if not apoif:
		p = state['air_pressure_on_interface_levels'].to_units('Pa').values
		state['air_pressure'] = make_dataarray_3d(
			0.5*(p[:, :, :-1] + p[:, :, 1:]), grid, 'Pa', name='air_pressure'
		)
		exn = state['exner_function_on_interface_levels'].to_units('J kg^-1 K^-1').values
		state['exner_function'] = make_dataarray_3d(
			0.5*(exn[:, :, :-1] + exn[:, :, 1:]), grid, 'J kg^-1 K^-1', name='exner_function'
		)

	dtype = grid.x.dtype

	rd = 287.0
	rv = 461.5
	lhvw = 2.5e6
	beta = rd / rv

	#
	# test properties
	#
	kessler = Kessler(
		domain, grid_type,
		air_pressure_on_interface_levels=apoif,
		tendency_of_air_potential_temperature_in_diagnostics=toaptid,
		rain_evaporation=re,
		autoconversion_threshold=DataArray(a, attrs={'units': 'g g^-1'}),
		autoconversion_rate=DataArray(k1, attrs={'units': 's^-1'}),
		collection_rate=DataArray(k2, attrs={'units': 'hr^-1'}),
		saturation_water_vapor_formula=swvf_type,
		backend=backend, dtype=dtype
	)

	assert 'air_density' in kessler.input_properties
	assert 'air_temperature' in kessler.input_properties
	assert mfwv in kessler.input_properties
	assert mfcw in kessler.input_properties
	assert mfpw in kessler.input_properties
	if apoif:
		assert 'air_pressure_on_interface_levels' in kessler.input_properties
		assert 'exner_function_on_interface_levels' in kessler.input_properties
	else:
		assert 'air_pressure' in kessler.input_properties
		assert 'exner_function' in kessler.input_properties
	assert len(kessler.input_properties) == 7

	tendency_names = []
	assert mfcw in kessler.tendency_properties
	tendency_names.append(mfcw)
	assert mfpw in kessler.tendency_properties
	tendency_names.append(mfpw)
	if re:
		assert mfwv in kessler.tendency_properties
		tendency_names.append(mfwv)
		if not toaptid:
			assert 'air_potential_temperature' in kessler.tendency_properties
			tendency_names.append('air_potential_temperature')
			assert len(kessler.tendency_properties) == 4
		else:
			assert len(kessler.tendency_properties) == 3
	else:
		assert len(kessler.tendency_properties) == 2

	diagnostic_names = []
	if re and toaptid:
		assert 'tendency_of_air_potential_temperature' in kessler.diagnostic_properties
		diagnostic_names.append('tendency_of_air_potential_temperature')
		assert len(kessler.diagnostic_properties) == 1
	else:
		assert len(kessler.diagnostic_properties) == 0

	#
	# test numerics
	#
	tendencies, diagnostics = kessler(state)

	for name in tendency_names:
		assert name in tendencies
	assert len(tendencies) == len(tendency_names)

	for name in diagnostic_names:
		assert name in diagnostics
	assert len(diagnostics) == len(diagnostic_names)

	rho = state['air_density'].to_units('kg m^-3').values
	p = state['air_pressure_on_interface_levels'].to_units('Pa').values \
		if apoif else state['air_pressure'].to_units('Pa').values
	t = state['air_temperature'].to_units('K').values
	exn = state['exner_function_on_interface_levels'].to_units('J kg^-1 K^-1').values \
		if apoif else state['exner_function'].to_units('J kg^-1 K^-1').values
	qv = state[mfwv].to_units('g g^-1').values
	qc = state[mfcw].to_units('g g^-1').values
	qr = state[mfpw].to_units('g g^-1').values

	assert kessler._a == a
	assert kessler._k1 == k1
	assert np.isclose(kessler._k2, k2/3600.0)

	swvf = goff_gratch_formula if swvf_type == 'goff_gratch' else tetens_formula

	tnd_qv, tnd_qc, tnd_qr, tnd_theta = kessler_validation(
		rho, p, t, exn, qv, qc, qr, a, k1, k2/3600.0, swvf, beta, lhvw, re
	)

	compare_dataarrays(
		make_dataarray_3d(tnd_qc, grid, 'g g^-1 s^-1'), tendencies[mfcw],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(tnd_qr, grid, 'g g^-1 s^-1'), tendencies[mfpw],
		compare_coordinate_values=False
	)
	if mfwv in tendency_names:
		compare_dataarrays(
			make_dataarray_3d(tnd_qv, grid, 'g g^-1 s^-1'), tendencies[mfwv],
			compare_coordinate_values=False
		)
	if 'air_potential_temperature' in tendency_names:
		compare_dataarrays(
			make_dataarray_3d(tnd_theta, grid, 'K s^-1'),
			tendencies['air_potential_temperature'],
			compare_coordinate_values=False
		)
	if 'tendency_of_air_potential_temperature' in diagnostic_names:
		compare_dataarrays(
			make_dataarray_3d(tnd_theta, grid, 'K s^-1'),
			diagnostics['tendency_of_air_potential_temperature'],
			compare_coordinate_values=False
		)


def saturation_adjustment_kessler_validation(p, t, qv, qc, beta, lhvw, cp):
	p = p if p.shape[2] == t.shape[2] else 0.5*(p[:, :, :-1] + p[:, :, 1:])
	pvs = tetens_formula(t)
	qvs = beta * pvs / (p - pvs)
	d = np.minimum((qvs - qv) / (1.0 + qvs * 4093 * lhvw / (cp * (t - 36)**2)), qc)
	return qv+d, qc-d


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_saturation_adjustment_kessler(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

	apoif = data.draw(hyp_st.booleans(), label="apoif")

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	if not apoif:
		p = state['air_pressure_on_interface_levels'].to_units('Pa').values
		state['air_pressure'] = make_dataarray_3d(
			0.5*(p[:, :, :-1] + p[:, :, 1:]), grid, 'Pa', name='air_pressure'
		)

	dtype = grid.x.dtype

	rd = 287.0
	rv = 461.5
	cp = 1004.0
	lhvw = 2.5e6
	beta = rd / rv

	#
	# test properties
	#
	sak = SaturationAdjustmentKessler(
		domain, grid_type, air_pressure_on_interface_levels=apoif,
		backend=backend, dtype=dtype
	)

	assert 'air_temperature' in sak.input_properties
	assert mfwv in sak.input_properties
	assert mfcw in sak.input_properties
	if apoif:
		assert 'air_pressure_on_interface_levels' in sak.input_properties
	else:
		assert 'air_pressure' in sak.input_properties
	assert len(sak.input_properties) == 4

	assert mfwv in sak.diagnostic_properties
	assert mfcw in sak.diagnostic_properties
	assert len(sak.diagnostic_properties) == 2

	#
	# test numerics
	#
	diagnostics = sak(state)

	assert mfwv in diagnostics
	assert mfcw in diagnostics
	assert len(diagnostics) == 2

	p = state['air_pressure_on_interface_levels'].to_units('Pa').values \
		if apoif else state['air_pressure'].to_units('Pa').values
	t = state['air_temperature'].to_units('K').values
	qv = state[mfwv].to_units('g g^-1').values
	qc = state[mfcw].to_units('g g^-1').values

	out_qv, out_qc = saturation_adjustment_kessler_validation(p, t, qv, qc, beta, lhvw, cp)

	compare_dataarrays(
		make_dataarray_3d(out_qv, grid, 'g g^-1'), diagnostics[mfwv],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(out_qc, grid, 'g g^-1'), diagnostics[mfcw],
		compare_coordinate_values=False
	)


def raindrop_fall_velocity_validation(rho, qr):
	return 36.34 * (0.001 * rho * qr)**0.1346 * np.sqrt(rho[:, :, -1:] / rho)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_raindrop_fall_velocity(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	dtype = grid.x.dtype

	#
	# test properties
	#
	rfv = RaindropFallVelocity(domain, grid_type, backend=backend, dtype=dtype)

	assert 'air_density' in rfv.input_properties
	assert mfpw in rfv.input_properties
	assert len(rfv.input_properties) == 2

	assert 'raindrop_fall_velocity' in rfv.diagnostic_properties
	assert len(rfv.diagnostic_properties) == 1

	#
	# test numerics
	#
	diagnostics = rfv(state)

	assert 'raindrop_fall_velocity' in diagnostics
	assert len(diagnostics) == 1

	rho = state['air_density'].to_units('kg m^-3').values
	qr = state[mfpw].to_units('g g^-1').values

	vt = raindrop_fall_velocity_validation(rho, qr)

	compare_dataarrays(
		make_dataarray_3d(vt, grid, 'm s^-1'), diagnostics['raindrop_fall_velocity'],
		compare_coordinate_values=False
	)


class WrappingStencil:
	def __init__(self, core, rho, h, qr, vt, backend):
		self._core = core

		nx, ny, nz = rho.shape
		dtype = rho.dtype
		self.out = np.zeros((nx, ny, nz), dtype=dtype)

		stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs={'rho': rho, 'h': h, 'qr': qr, 'vt': vt},
			outputs={'tmp_dfdz': self.out},
			domain=gt.domain.Rectangle((0, 0, self._core.nb), (nx-1, ny-1, nz-1)),
			mode=backend
		)

		stencil.compute()

	def stencil_defs(self, rho, h, qr, vt):
		k = gt.Index(axis=2)
		return self._core(k, rho, h, qr, vt)


def first_order_flux_validation(rho, h, qr, vt):
	tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

	out = np.zeros_like(rho, dtype=rho.dtype)
	out[:, :, 1:] = \
		(
			rho[:, :, :-1] * qr[:, :, :-1] * vt[:, :, :-1] -
			rho[:, :, 1: ] * qr[:, :, 1: ] * vt[:, :, 1: ]
		) / (tmp_h[:, :, :-1] - tmp_h[:, :, 1:])

	return out


def second_order_flux_validation(rho, h, qr, vt):
	tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

	a = np.zeros_like(rho, dtype=rho.dtype)
	a[:, :, 2:] = (2*tmp_h[:, :, 2:] - tmp_h[:, :, 1:-1] - tmp_h[:, :, :-2]) / \
		((tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]))
	b = np.zeros_like(rho, dtype=rho.dtype)
	b[:, :, 2:] = (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) / \
		((tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1]))
	c = np.zeros_like(rho, dtype=rho.dtype)
	c[:, :, 2:] = - (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) / \
		((tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1]))

	out = np.zeros_like(rho, dtype=rho.dtype)
	out[:, :, 2:] = \
		a[:, :, 2:] * (rho[:, :, 2:  ] * qr[:, :, 2:  ] * vt[:, :, 2:  ]) + \
		b[:, :, 2:] * (rho[:, :, 1:-1] * qr[:, :, 1:-1] * vt[:, :, 1:-1]) + \
		c[:, :, 2:] * (rho[:, :, :-2 ] * qr[:, :, :-2 ] * vt[:, :, :-2 ])

	return out


flux_properties = {
	'first_order_upwind': {
		'type': _FirstOrderUpwind, 'validation': first_order_flux_validation
	},
	'second_order_upwind': {
		'type': _SecondOrderUpwind, 'validation': second_order_flux_validation
	}
}


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_sedimentation_flux(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	dtype = grid.x.dtype

	rho = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label="rho"
	)
	h = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz+1),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label="h"
	)
	qr = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label="qr"
	)
	vt = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label="vt"
	)

	flux_type = data.draw(
		st_one_of(('first_order_upwind', 'second_order_upwind')),
		label="flux_type"
	)

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	core = SedimentationFlux.factory(flux_type)
	assert isinstance(core, flux_properties[flux_type]['type'])
	ws = WrappingStencil(core, rho, h, qr, vt, backend)
	assert np.allclose(
		ws.out, flux_properties[flux_type]['validation'](rho, h, qr, vt),
		equal_nan=True
	)


def sedimentation_validation(state, timestep, flux_type, maxcfl):
	rho = state['air_density'].to_units('kg m^-3').values
	h = state['height_on_interface_levels'].to_units('m').values
	qr = state[mfpw].to_units('g g^-1').values
	vt = state['raindrop_fall_velocity'].to_units('m s^-1').values

	dt = timestep.total_seconds()
	dh = h[:, :, :-1] - h[:, :, 1:]
	ids = np.where(vt > maxcfl * dh / dt)
	vt[ids] = maxcfl * dh[ids] / dt

	dfdz = flux_properties[flux_type]['validation'](rho, h, qr, vt)

	return dfdz / rho


def precipitation_validation(state, timestep, maxcfl, rhow):
	rho = state['air_density'].to_units('kg m^-3').values
	h = state['height_on_interface_levels'].to_units('m').values
	qr = state[mfpw].to_units('g g^-1').values
	vt = state['raindrop_fall_velocity'].to_units('m s^-1').values

	dt = timestep.total_seconds()
	dh = h[:, :, :-1] - h[:, :, 1:]
	ids = np.where(vt > maxcfl * dh / dt)
	vt[ids] = maxcfl * dh[ids] / dt

	return 3.6e6 * rho[:, :, -1:] * qr[:, :, -1:] * vt[:, :, -1:] / rhow


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_sedimentation(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

	flux_type = data.draw(
		st_one_of(('first_order_upwind', 'second_order_upwind')),
		label="flux_type"
	)
	maxcfl = data.draw(hyp_st.floats(min_value=0, max_value=1), label="maxcfl")

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
		),
		label="timestep"
	)

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	dtype = grid.x.dtype

	rfv = RaindropFallVelocity(domain, grid_type, backend=backend, dtype=dtype)
	diagnostics = rfv(state)
	state.update(diagnostics)

	sed = Sedimentation(
		domain, grid_type, flux_type, maxcfl,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	#
	# test properties
	#
	assert 'air_density' in sed.input_properties
	assert 'height_on_interface_levels' in sed.input_properties
	assert mfpw in sed.input_properties
	assert 'raindrop_fall_velocity' in sed.input_properties
	assert len(sed.input_properties) == 4

	assert mfpw in sed.tendency_properties
	assert len(sed.tendency_properties) == 1

	assert len(sed.diagnostic_properties) == 0

	#
	# test numerics
	#
	tendencies, diagnostics = sed(state, timestep)

	assert mfpw in tendencies
	raw_mfpw_val = sedimentation_validation(state, timestep, flux_type, maxcfl)
	compare_dataarrays(
		make_dataarray_3d(raw_mfpw_val, grid, 'g g^-1 s^-1'),
		tendencies[mfpw], compare_coordinate_values=False
	)
	assert len(tendencies) == 1

	assert len(diagnostics) == 0


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_precipitation(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label="domain")

	grid_type = data.draw(st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state = data.draw(st_isentropic_state_f(grid, moist=True, precipitation=True), label="state")

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
		),
		label="timestep"
	)

	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	dtype = grid.x.dtype

	rfv = RaindropFallVelocity(domain, grid_type, backend=backend, dtype=dtype)
	state.update(rfv(state))

	comp = Precipitation(domain, grid_type, backend=backend, dtype=dtype)

	tendencies, diagnostics = comp(state, timestep)

	assert len(tendencies) == 0

	rho = state['air_density'].to_units('kg m^-3').values[:, :, -1:]
	qr = state[mfpw].to_units('g g^-1').values[:, :, -1:]
	vt = state['raindrop_fall_velocity'].to_units('m s^-1').values[:, :, -1:]
	rhow = comp._rhow.value
	prec = 3.6e6 * rho * qr * vt / rhow
	assert 'precipitation' in diagnostics
	compare_dataarrays(
		make_dataarray_3d(prec, grid, 'mm hr^-1'),
		diagnostics['precipitation'], compare_coordinate_values=False
	)

	accprec = state['accumulated_precipitation'].to_units('mm').values
	accprec_val = accprec + timestep.total_seconds() * prec / 3.6e3
	assert 'accumulated_precipitation' in diagnostics
	compare_dataarrays(
		make_dataarray_3d(accprec_val, grid, 'mm'),
		diagnostics['accumulated_precipitation'], compare_coordinate_values=False
	)

	assert len(diagnostics) == 2


if __name__ == '__main__':
	pytest.main([__file__])
