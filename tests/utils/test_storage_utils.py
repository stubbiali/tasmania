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
from datetime import datetime, timedelta
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    seed,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest
import tempfile

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from tasmania import get_dataarray_3d
from tasmania.python.utils.io_utils import NetCDFMonitor, load_netcdf_dataset


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_store_pp(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    pgrid = domain.physical_grid
    pstate = data.draw(
        utils.st_isentropic_state(pgrid, moist=True, precipitation=True), label="pstate"
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "physical")
    netcdf.store(pstate)
    netcdf.store(pstate)
    netcdf.store(pstate)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,  # HealthCheck.filter_too_much
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_store_pc(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    cgrid = domain.numerical_grid
    cstate = data.draw(
        utils.st_isentropic_state(cgrid, moist=False, precipitation=False), label="cstate"
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "physical")
    netcdf.store(cstate)
    netcdf.store(cstate)
    netcdf.store(cstate)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_store_cc(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    cgrid = domain.physical_grid
    cstate = data.draw(
        utils.st_isentropic_state(cgrid, moist=True, precipitation=True), label="cstate"
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "numerical")
    netcdf.store(cstate)
    netcdf.store(cstate)
    netcdf.store(cstate)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,  # HealthCheck.filter_too_much
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_store_cp(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    pgrid = domain.numerical_grid
    pstate = data.draw(
        utils.st_isentropic_state(pgrid, moist=False, precipitation=False), label="pstate"
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "numerical")
    netcdf.store(pstate)
    netcdf.store(pstate)
    netcdf.store(pstate)


def assert_grids(g1, g2):
    # x-axis
    assert g1.nx == g2.nx
    assert g1.dx == g2.dx
    utils.compare_dataarrays(g1.x, g2.x)
    utils.compare_dataarrays(g1.x_at_u_locations, g2.x_at_u_locations)

    # y-axis
    assert g1.ny == g2.ny
    assert g1.dy == g2.dy
    utils.compare_dataarrays(g1.y, g2.y)
    utils.compare_dataarrays(g1.y_at_v_locations, g2.y_at_v_locations)

    # z-axis
    assert g1.nz == g2.nz
    assert g1.dz == g2.dz
    utils.compare_dataarrays(g1.z, g2.z)
    utils.compare_dataarrays(g1.z_on_interface_levels, g2.z_on_interface_levels)

    # topography
    topo1, topo2 = g1.topography, g2.topography
    assert topo1.type == topo2.type
    assert np.allclose(topo1.steady_profile, topo2.steady_profile)


def assert_isentropic_states(state, state_ref):
    assert len(state) == len(state_ref)

    for name in state_ref:
        assert name in state

        if name == "time":
            assert state["time"] == state_ref["time"]
        else:
            assert np.allclose(state[name], state_ref[name])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_write_and_load(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    assume(domain.horizontal_boundary.type != "dirichlet")

    pgrid = domain.physical_grid
    cgrid = domain.numerical_grid

    pstate = data.draw(
        utils.st_isentropic_state(
            pgrid,
            moist=False,
            precipitation=False,
            time=datetime(year=1992, month=2, day=20),
        ),
        label="pstate",
    )

    hb = domain.horizontal_boundary
    cstate = {"time": pstate["time"]}
    for name in pstate:
        if name != "time":
            pfield = pstate[name].values
            units = pstate[name].attrs["units"]
            cfield = hb.get_numerical_field(pfield, field_name=name)
            cstate[name] = get_dataarray_3d(cfield, cgrid, units, name=name)

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    _, filename = tempfile.mkstemp(suffix=".nc", dir="tmp")
    os.remove(filename)

    # ========================================
    # test bed
    # ========================================
    # instantiate the monitor
    netcdf = NetCDFMonitor(filename, domain, "physical")

    # store the states
    netcdf.store(cstate)
    cstate["time"] += timedelta(hours=1)
    netcdf.store(cstate)
    cstate["time"] += timedelta(hours=1)
    netcdf.store(cstate)

    # dump to file
    netcdf.write()

    # retrieve data from file
    load_domain, load_grid_type, load_states = load_netcdf_dataset(filename)

    # physical grid
    assert_grids(domain.physical_grid, load_domain.physical_grid)

    # numerical grid
    assert_grids(domain.numerical_grid, load_domain.numerical_grid)

    # underlying grid type
    assert load_grid_type == "physical"

    # states
    assert len(load_states) == 3
    for state in load_states:
        assert_isentropic_states(state, pstate)
        pstate["time"] += timedelta(hours=1)

    # clean temporary directory
    os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__])