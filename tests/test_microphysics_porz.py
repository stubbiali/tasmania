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
from tasmania.python.physics.microphysics.porz import \
	PorzMicrophysics, PorzFallVelocity, PorzSedimentation
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.meteo_utils import \
	goff_gratch_formula, tetens_formula

try:
	from .conf import backend as conf_backend
	from .test_microphysics_utils import porz_sedimentation_validation
	from .utils import compare_arrays, compare_dataarrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f
except ModuleNotFoundError:
	from conf import backend as conf_backend
	from test_microphysics_utils import porz_sedimentation_validation
	from utils import compare_arrays, compare_dataarrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'
ndpw = 'number_density_of_precipitation_water'


def porz_microphysics_validation(
	rho, p, ps, T, qv, qc, qr, nr, Ninf, pref, rhol, Rd, Rv, lhwv, cp, rain_evaporation
):
	ae       = 0.78
	alpha    = 190.3
	ak       = 0.002646
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
	T0       = 273.15
	T_mu     = 110.4

	p = p if p.shape[2] == rho.shape[2] else 0.5*(p[:, :, :-1] + p[:, :, 1:])

	D = D0 * (T / T0)**1.94 * p_star / p
	K = ak * T**1.5 / (T + bk * 10**(ck / T))
	G = 1.0 / ((lhwv / (Rv * T) - 1.0) * lhwv * ps * D / (Rv * T**2 * K) + 1.0)
	d = 4.0 * np.pi * (3.0 / (4.0 * np.pi * rhol))**(1.0 / 3.0) * D * G
	mu = mu0 * T**1.5 / (T + T_mu)
	be = bv * (mu / (rho * D))**(1.0 / 3.0) * (2.0 * rho / mu)**0.5 * \
		(3.0 / (4.0 * np.pi * rhol))**(1.0 / 6.0)

	qvs = eps * ps / p
	vt = alpha * qr**beta * (mt / (qr + mt * nr))**beta * (rho_star / rho)**0.5
	nc = qc * Ninf / (qc + Ninf * m0) / np.tanh(qc / (N0 * m0))

	A1 = k1 * rho * qc**2 / rhol
	A1p = 0.5 * k1 * rho * nc * qc / rhol
	A2 = k2 * np.pi * (3.0 / (4.0 * np.pi * rhol))**(2.0 / 3.0) * vt * rho * \
		qc * qr**(2.0 / 3.0) * nr**(1.0 / 3.0)
	C = d * rho * (qv - qvs) * nc**(2.0 / 3.0) * qc**(1.0 / 3.0)

	if rain_evaporation:
		E = d * rho * ((qvs - qv) > 0.0) * (qvs - qv) * \
			(ae * qr**(1.0 / 3.0) * nr**(2.0 / 3.0) +
			 be * vt**0.5 * qr**0.5 * nr**0.5)
		Ep = E * nr / qr
		Ep[qr <= 0.0] = 0.0

		tnd_qv = - C + E
		tnd_qc = C - A1 - A2
		tnd_qr = A1 + A2 - E
		tnd_nr = A1p - Ep
		tnd_theta = (pref / p)**(Rd / cp) * lhwv * (C - E) / cp
	else:
		tnd_qv = - C
		tnd_qc = C - A1 - A2
		tnd_qr = A1 + A2
		tnd_nr = A1p
		tnd_theta = (pref / p)**(Rd / cp) * lhwv * C / cp

	return tnd_qv, tnd_qc, tnd_qr, tnd_nr, tnd_theta


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_porz_microphysics(data):
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

	Ninf = data.draw(st_floats(min_value=1, max_value=1e9), label="Ninf")

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

	#
	# test properties
	#
	porz = PorzMicrophysics(
		domain, grid_type,
		air_pressure_on_interface_levels=apoif,
		tendency_of_air_potential_temperature_in_diagnostics=toaptid,
		rain_evaporation=re, saturation_water_vapor_formula=swvf_type,
		activation_parameter=DataArray(Ninf, attrs={'units': 'kg^-1'}),
		backend=backend, dtype=dtype
	)

	assert 'air_density' in porz.input_properties
	assert 'air_temperature' in porz.input_properties
	assert mfwv in porz.input_properties
	assert mfcw in porz.input_properties
	assert mfpw in porz.input_properties
	assert ndpw in porz.input_properties
	if apoif:
		assert 'air_pressure_on_interface_levels' in porz.input_properties
	else:
		assert 'air_pressure' in porz.input_properties
	assert len(porz.input_properties) == 7

	tendency_names = [mfwv, mfcw, mfpw, ndpw]
	diagnostic_names = []
	if toaptid:
		diagnostic_names.append('tendency_of_air_potential_temperature')
	else:
		tendency_names.append('air_potential_temperature')

	for tendency_name in tendency_names:
		assert tendency_name in porz.tendency_properties
	assert len(porz.tendency_properties) == len(tendency_names)

	for diagnostic_name in diagnostic_names:
		assert diagnostic_name in porz.diagnostic_properties
	assert len(porz.diagnostic_properties) == len(diagnostic_names)

	#
	# test numerics
	#
	tendencies, diagnostics = porz(state)

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
	qv = state[mfwv].to_units('g g^-1').values
	qc = state[mfcw].to_units('g g^-1').values
	qr = state[mfpw].to_units('g g^-1').values
	nr = state[ndpw].to_units('kg^-1').values

	swvf = goff_gratch_formula if swvf_type == 'goff_gratch' else tetens_formula
	ps = swvf(t)

	pref = porz._physical_constants['air_pressure_at_sea_level']
	rhol = porz._physical_constants['density_of_liquid_water']
	rd   = porz._physical_constants['gas_constant_of_dry_air']
	rv   = porz._physical_constants['gas_constant_of_water_vapor']
	lhwv = porz._physical_constants['latent_heat_of_vaporization_of_water']
	cp   = porz._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

	tnd_qv, tnd_qc, tnd_qr, tnd_nr, tnd_theta = porz_microphysics_validation(
		rho, p, ps, t, qv, qc, qr, nr, Ninf, pref, rhol, rd, rv, lhwv, cp, re
	)

	compare_dataarrays(
		make_dataarray_3d(tnd_qv, grid, 'kg kg^-1 s^-1'), tendencies[mfwv],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(tnd_qc, grid, 'kg kg^-1 s^-1'), tendencies[mfcw],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(tnd_qr, grid, 'kg kg^-1 s^-1'), tendencies[mfpw],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(tnd_nr, grid, 'kg^-1 s^-1'), tendencies[ndpw],
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


def porz_fall_velocity_validation(rho, qr, nr):
	alpha    = 190.3
	beta     = 4.0/15.0
	cn       = 0.58
	cq       = 1.84
	mt       = 1.21e-5
	rho_star = 1.225

	vt = alpha * qr**beta * (mt / (qr + mt * nr))**beta * (rho_star / rho)**0.5
	vq = cq * vt
	vn = cn * vt

	return vq, vn



@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_porz_fall_velocity(data):
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
	porz = PorzFallVelocity(domain, grid_type, backend=backend, dtype=dtype)

	assert 'air_density' in porz.input_properties
	assert mfpw in porz.input_properties
	assert ndpw in porz.input_properties
	assert len(porz.input_properties) == 3

	assert 'raindrop_fall_velocity' in porz.diagnostic_properties
	assert 'number_density_of_raindrop_fall_velocity' in porz.diagnostic_properties
	assert len(porz.diagnostic_properties) == 2

	#
	# test numerics
	#
	diagnostics = porz(state)

	assert 'raindrop_fall_velocity' in porz.diagnostic_properties
	assert 'number_density_of_raindrop_fall_velocity' in porz.diagnostic_properties
	assert len(diagnostics) == 2

	rho = state['air_density'].to_units('kg m^-3').values
	qr = state[mfpw].to_units('g g^-1').values
	nr = state[ndpw].to_units('kg^-1').values

	vq, vn = porz_fall_velocity_validation(rho, qr, nr)

	compare_dataarrays(
		make_dataarray_3d(vq, grid, 'm s^-1'),
		diagnostics['raindrop_fall_velocity'],
		compare_coordinate_values=False
	)
	compare_dataarrays(
		make_dataarray_3d(vn, grid, 'm s^-1'),
		diagnostics['number_density_of_raindrop_fall_velocity'],
		compare_coordinate_values=False
	)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_porz_sedimentation(data):
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

	rfv = PorzFallVelocity(domain, grid_type, backend=backend, dtype=dtype)
	diagnostics = rfv(state)
	state.update(diagnostics)

	sed = PorzSedimentation(
		domain, grid_type, flux_type, maxcfl,
		backend=gt.mode.NUMPY, dtype=dtype
	)

	#
	# test properties
	#
	assert 'air_density' in sed.input_properties
	assert 'height_on_interface_levels' in sed.input_properties
	assert mfpw in sed.input_properties
	assert ndpw in sed.input_properties
	assert 'raindrop_fall_velocity' in sed.input_properties
	assert 'number_density_of_raindrop_fall_velocity' in sed.input_properties
	assert len(sed.input_properties) == 6

	assert mfpw in sed.tendency_properties
	assert ndpw in sed.tendency_properties
	assert len(sed.tendency_properties) == 2

	assert len(sed.diagnostic_properties) == 0

	#
	# test numerics
	#
	tendencies, diagnostics = sed(state, timestep)

	raw_mfpw_val, raw_ndpw_val = porz_sedimentation_validation(
		state, timestep, flux_type, maxcfl
	)

	assert mfpw in tendencies
	compare_dataarrays(
		make_dataarray_3d(raw_mfpw_val, grid, 'g g^-1 s^-1'),
		tendencies[mfpw], compare_coordinate_values=False
	)
	assert ndpw in tendencies
	compare_dataarrays(
		make_dataarray_3d(raw_ndpw_val, grid, 'kg^-1 s^-1'),
		tendencies[ndpw], compare_coordinate_values=False
	)
	assert len(tendencies) == 2

	assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
