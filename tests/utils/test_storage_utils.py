# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
    reproduce_failure,
    strategies as hyp_st,
)
import os
import pytest
import tempfile

from tasmania import get_dataarray_3d
from tasmania.python.utils.io import NetCDFMonitor, load_netcdf_dataset

from tests.strategies import st_domain, st_isentropic_state
from tests.utilities import compare_arrays, compare_dataarrays, hyp_settings


@hyp_settings
@given(hyp_st.data())
def test_store_pp(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    pgrid = domain.physical_grid
    pstate = data.draw(
        st_isentropic_state(pgrid, moist=True, precipitation=True),
        label="pstate",
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "physical")
    netcdf.store(pstate)
    netcdf.store(pstate)
    netcdf.store(pstate)


@hyp_settings
@given(hyp_st.data())
def test_store_pn(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    ngrid = domain.numerical_grid
    nstate = data.draw(
        st_isentropic_state(ngrid, moist=False, precipitation=False),
        label="cstate",
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "physical")
    netcdf.store(nstate)
    netcdf.store(nstate)
    netcdf.store(nstate)


@hyp_settings
@given(hyp_st.data())
def test_store_nn(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    ngrid = domain.numerical_grid
    nstate = data.draw(
        st_isentropic_state(ngrid, moist=True, precipitation=True),
        label="cstate",
    )

    filename = data.draw(hyp_st.text(), label="filename")

    # ========================================
    # test bed
    # ========================================
    netcdf = NetCDFMonitor(filename, domain, "numerical")
    netcdf.store(nstate)
    netcdf.store(nstate)
    netcdf.store(nstate)


@hyp_settings
@given(hyp_st.data())
def test_store_np(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    pgrid = domain.physical_grid
    pstate = data.draw(
        st_isentropic_state(pgrid, moist=False, precipitation=False),
        label="pstate",
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
    compare_dataarrays(g1.x, g2.x)
    compare_dataarrays(g1.x_at_u_locations, g2.x_at_u_locations)

    # y-axis
    assert g1.ny == g2.ny
    assert g1.dy == g2.dy
    compare_dataarrays(g1.y, g2.y)
    compare_dataarrays(g1.y_at_v_locations, g2.y_at_v_locations)

    # z-axis
    assert g1.nz == g2.nz
    assert g1.dz == g2.dz
    compare_dataarrays(g1.z, g2.z)
    compare_dataarrays(g1.z_on_interface_levels, g2.z_on_interface_levels)

    # topography
    topo1, topo2 = g1.topography, g2.topography
    assert topo1.type == topo2.type
    compare_dataarrays(topo1.steady_profile, topo2.steady_profile)


def assert_isentropic_states(state, state_ref, *, subtests):
    assert len(state) == len(state_ref)

    for name in state_ref:
        # with subtests.test(name=name):
        assert name in state

        if name == "time":
            assert state["time"] == state_ref["time"]
        else:
            compare_dataarrays(
                state[name], state_ref[name], compare_coordinate_values=False
            )


@hyp_settings
@given(data=hyp_st.data())
def test_write_and_load(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    # assume(domain.horizontal_boundary.type != "dirichlet")

    pgrid = domain.physical_grid
    ngrid = domain.numerical_grid

    pstate = data.draw(
        st_isentropic_state(
            pgrid,
            moist=False,
            precipitation=False,
            time=datetime(year=1992, month=2, day=20),
        ),
        label="pstate",
    )

    hb = domain.horizontal_boundary
    nstate = {"time": pstate["time"]}
    for name in pstate:
        if name != "time":
            pfield = pstate[name].values
            units = pstate[name].attrs["units"]
            nfield = hb.get_numerical_field(pfield, field_name=name)
            nstate[name] = get_dataarray_3d(nfield, ngrid, units, name=name)

    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")

    _, filename = tempfile.mkstemp(suffix=".nc", dir=".tmp")
    os.remove(filename)

    # ========================================
    # test bed
    # ========================================
    # instantiate the monitor
    netcdf = NetCDFMonitor(filename, domain, "physical")

    # store the states
    netcdf.store(nstate)
    nstate["time"] += timedelta(hours=1)
    netcdf.store(nstate)
    nstate["time"] += timedelta(hours=1)
    netcdf.store(nstate)

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
    for idx, state in enumerate(load_states):
        # with subtests.test(idx=idx):
        assert_isentropic_states(state, pstate, subtests=subtests)
        pstate["time"] += timedelta(hours=1)

    # clean temporary directory
    os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__])
