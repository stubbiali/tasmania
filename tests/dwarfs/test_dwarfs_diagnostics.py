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
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import pytest

import gt4py as gt

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent
from tasmania.python.utils.storage_utils import zeros

from tests.conf import (
    backend as conf_backend,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utilities import (
    compare_arrays,
    st_floats,
    st_one_of,
    st_domain,
    st_raw_field,
)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_horizontal_velocity_staggered(data):
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 20), nb=1),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    r = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )
    u = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )
    v = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )

    # ========================================
    # test bed
    # ========================================
    ru = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
    rv = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

    hv = HorizontalVelocity(grid, True, backend=backend, dtype=dtype, rebuild=False)

    hv.get_momenta(r, u, v, ru, rv)

    ru_val = r[:-1, :-1, :-1] * 0.5 * (u[:-1, :-1, :-1] + u[1:, :-1, :-1])
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * 0.5 * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
    v_new = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    u_new_val = (ru[:-2, :] + ru[1:-1, :]) / (r[:-2, :] + r[1:-1, :])
    compare_arrays(u_new[1:-1, :-1, :-1], u_new_val[:, :-1, :-1])
    v_new_val = (rv[:, :-2] + rv[:, 1:-1]) / (r[:, :-2] + r[:, 1:-1])
    compare_arrays(v_new[:-1, 1:-1, :-1], v_new_val[:-1, :, :-1])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_horizontal_velocity(data):
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 20), nb=1),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    r = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )
    u = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )
    v = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )

    # ========================================
    # test bed
    # ========================================
    ru = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
    rv = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

    hv = HorizontalVelocity(grid, False, backend=backend, dtype=dtype, rebuild=False)

    hv.get_momenta(r, u, v, ru, rv)

    ru_val = r[:-1, :-1, :-1] * u[:-1, :-1, :-1]
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * v[:-1, :-1, :-1]
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
    v_new = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    u_new_val = ru[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(u_new[:-1, :-1, :-1], u_new_val)
    v_new_val = rv[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(v_new[:-1, :-1, :-1], v_new_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_water_constituent(data):
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 20), nb=1),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    r = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )
    q = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="r",
    )

    # ========================================
    # test bed
    # ========================================
    rq = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
    q_new = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

    #
    # clipping off
    #
    wc = WaterConstituent(grid, False, backend=backend, dtype=dtype, rebuild=False)

    wc.get_density_of_water_constituent(r, q, rq)
    rq_val = r[:-1, :-1, :-1] * q[:-1, :-1, :-1]
    compare_arrays(rq[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new)
    q_new_val = rq[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(q_new[:-1, :-1, :-1], q_new_val)

    #
    # clipping on
    #
    wc = WaterConstituent(grid, True, backend=backend, dtype=dtype, rebuild=False)

    wc.get_density_of_water_constituent(r, q, rq)
    rq_val = r[:-1, :-1, :-1] * q[:-1, :-1, :-1]
    rq_val[rq_val < 0.0] = 0.0
    compare_arrays(rq[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new)
    q_new_val = rq[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    q_new_val[q_new_val < 0.0] = 0.0
    compare_arrays(q_new[:-1, :-1, :-1], q_new_val)


if __name__ == "__main__":
    pytest.main([__file__])
