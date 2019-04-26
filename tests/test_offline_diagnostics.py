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
from hypothesis import \
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
import numpy as np
import pytest
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from tasmania.python.framework.offline_diagnostics import \
	FakeComponent, OfflineDiagnosticComponent, RMSD, RRMSD


units = {
	'air_density': ('kg m^-3', 'g cm^-3'),
	'air_isentropic_density': ('kg m^-2 K^-1', 'g km^-2 K^-1'),
	'air_pressure_on_interface_levels': ('Pa', 'kPa', 'atm'),
	'air_temperature': ('K', ),
	'exner_function_on_interface_levels': ('J kg^-1 K^-1', ),
	'height_on_interface_levels': ('m', 'km'),
	'montgomery_potential': ('m^2 s^-2', ),
	'x_momentum_isentropic': ('kg m^-1 K^-1 s^-1', ),
	'x_velocity_at_u_locations': ('m s^-1', 'km hr^-1'),
	'y_momentum_isentropic': ('kg m^-1 K^-1 s^-1', ),
	'y_velocity_at_v_locations': ('m s^-1', 'km hr^-1'),
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
def _test_fake_component(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")
	grid_type = data.draw(utils.st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid

	# ========================================
	# test bed
	# ========================================
	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	dims_stgx = (grid.x_at_u_locations.dims[0], grid.y.dims[0], grid.z.dims[0])
	dims_stgz = (grid.x.dims[0], grid.y.dims[0], grid.z_on_interface_levels.dims[0])

	properties = {
		'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
		'air_pressure_on_interface_levels': {'dims': dims_stgz, 'units': 'hPa'},
		'x_velocity_at_u_locations': {'dims': dims_stgx, 'units': 'km hr^-1'},
	}

	fc = FakeComponent({'input_properties': properties})
	assert hasattr(fc, 'input_properties')
	assert fc.input_properties == properties

	fc = FakeComponent({'tendency_properties': properties})
	assert hasattr(fc, 'tendency_properties')
	assert fc.tendency_properties == properties

	fc = FakeComponent({'diagnostic_properties': properties})
	assert hasattr(fc, 'diagnostic_properties')
	assert fc.diagnostic_properties == properties

	fc = FakeComponent({'output_properties': properties})
	assert hasattr(fc, 'output_properties')
	assert fc.output_properties == properties


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def _test_rmsd(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")
	grid_type = data.draw(utils.st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state_1 = data.draw(utils.st_isentropic_state_ff(grid, moist=True), label="state_1")
	state_2 = data.draw(utils.st_isentropic_state_ff(grid, moist=True), label="state_2")

	nfields = data.draw(hyp_st.integers(min_value=1, max_value=5), label="nfields")
	fields = {}
	for _ in range(nfields):
		field_name = data.draw(utils.st_one_of(units.keys()))
		field_units = data.draw(utils.st_one_of(units[field_name]))
		fields[field_name] = field_units

	xmin = data.draw(hyp_st.integers(min_value=0, max_value=grid.nx-1), label="xmin")
	xmax = data.draw(hyp_st.integers(min_value=xmin+1, max_value=grid.nx), label="xmax")
	ymin = data.draw(hyp_st.integers(min_value=0, max_value=grid.ny-1), label="ymin")
	ymax = data.draw(hyp_st.integers(min_value=ymin+1, max_value=grid.ny), label="ymax")
	zmin = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz-1), label="zmin")
	zmax = data.draw(hyp_st.integers(min_value=zmin+1, max_value=grid.nz), label="zmax")

	# ========================================
	# test bed
	# ========================================
	x = slice(xmin, xmax)
	y = slice(ymin, ymax)
	z = slice(zmin, zmax)

	rmsd = RMSD(grid, fields, x=x, y=y, z=z)
	assert isinstance(rmsd, OfflineDiagnosticComponent)

	diagnostics = rmsd(state_1, state_2)

	for field_name, field_units in fields.items():
		diag_name = 'rmsd_of_' + field_name
		assert diag_name in diagnostics

		val1 = state_1[field_name][x, y, z].to_units(field_units).values
		val2 = state_2[field_name][x, y, z].to_units(field_units).values
		val = np.sqrt(np.sum((val1 - val2)**2) / ((xmax-xmin)*(ymax-ymin)*(zmax-zmin)))
		diag_val = DataArray(
			np.array(val)[np.newaxis, np.newaxis, np.newaxis],
			dims=('scalar', 'scalar', 'scalar'), attrs={'units': field_units}
		)
		utils.compare_dataarrays(
			diagnostics[diag_name], diag_val, compare_coordinate_values=False
		)

	assert len(diagnostics) == len(fields)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_rrmsd(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")
	grid_type = data.draw(utils.st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	state_1 = data.draw(utils.st_isentropic_state_ff(grid, moist=True), label="state_1")
	state_2 = data.draw(utils.st_isentropic_state_ff(grid, moist=True), label="state_2")

	nfields = data.draw(hyp_st.integers(min_value=1, max_value=5), label="nfields")
	fields = {}
	for _ in range(nfields):
		field_name = data.draw(utils.st_one_of(units.keys()))
		field_units = data.draw(utils.st_one_of(units[field_name]))
		fields[field_name] = field_units

	xmin = data.draw(hyp_st.integers(min_value=0, max_value=grid.nx-1), label="xmin")
	xmax = data.draw(hyp_st.integers(min_value=xmin+1, max_value=grid.nx), label="xmax")
	ymin = data.draw(hyp_st.integers(min_value=0, max_value=grid.ny-1), label="ymin")
	ymax = data.draw(hyp_st.integers(min_value=ymin+1, max_value=grid.ny), label="ymax")
	zmin = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz-1), label="zmin")
	zmax = data.draw(hyp_st.integers(min_value=zmin+1, max_value=grid.nz), label="zmax")

	# ========================================
	# test bed
	# ========================================
	x = slice(xmin, xmax)
	y = slice(ymin, ymax)
	z = slice(zmin, zmax)

	rrmsd = RRMSD(grid, fields, x=x, y=y, z=z)
	assert isinstance(rrmsd, OfflineDiagnosticComponent)

	diagnostics = rrmsd(state_1, state_2)

	for field_name, field_units in fields.items():
		diag_name = 'rrmsd_of_' + field_name
		assert diag_name in diagnostics

		val1 = state_1[field_name][x, y, z].to_units(field_units).values
		val2 = state_2[field_name][x, y, z].to_units(field_units).values
		val = np.sqrt(np.sum((val1 - val2)**2) / np.sum(val2**2))
		diag_val = DataArray(
			np.array(val)[np.newaxis, np.newaxis, np.newaxis],
			dims=('scalar', 'scalar', 'scalar'), attrs={'units': '1'}
		)
		utils.compare_dataarrays(
			diagnostics[diag_name], diag_val, compare_coordinate_values=False
		)

	assert len(diagnostics) == len(fields)


if __name__ == '__main__':
	pytest.main([__file__])
