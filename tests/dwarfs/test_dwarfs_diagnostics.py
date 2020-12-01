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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.dwarfs.diagnostics import (
    HorizontalVelocity,
    WaterConstituent,
)
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.allocators import zeros

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_one_of, st_domain, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_horizontal_velocity_staggered(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

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
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    ru = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    rv = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv = HorizontalVelocity(
        grid, True, backend=backend, backend_options=bo, storage_options=so
    )

    hv.get_momenta(r, u, v, ru, rv)

    ru_val = r[:-1, :-1, :-1] * 0.5 * (u[:-1, :-1, :-1] + u[1:, :-1, :-1])
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * 0.5 * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    v_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    u_new_val = (ru[:-2, :] + ru[1:-1, :]) / (r[:-2, :] + r[1:-1, :])
    compare_arrays(u_new[1:-1, :-1, :-1], u_new_val[:, :-1, :-1])
    v_new_val = (rv[:, :-2] + rv[:, 1:-1]) / (r[:, :-2] + r[:, 1:-1])
    compare_arrays(v_new[:-1, 1:-1, :-1], v_new_val[:-1, :, :-1])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_horizontal_velocity(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

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
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    ru = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    rv = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv = HorizontalVelocity(
        grid, False, backend=backend, backend_options=bo, storage_options=so
    )

    hv.get_momenta(r, u, v, ru, rv)

    ru_val = r[:-1, :-1, :-1] * u[:-1, :-1, :-1]
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * v[:-1, :-1, :-1]
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    v_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    u_new_val = ru[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(u_new[:-1, :-1, :-1], u_new_val)
    v_new_val = rv[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(v_new[:-1, :-1, :-1], v_new_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_water_constituent(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

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
        label="q",
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    rq = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    q_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    #
    # clipping off
    #
    wc = WaterConstituent(
        grid, False, backend=backend, backend_options=bo, storage_options=so
    )

    wc.get_density_of_water_constituent(r, q, rq)
    rq_val = r[:-1, :-1, :-1] * q[:-1, :-1, :-1]
    compare_arrays(rq[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new)
    q_new_val = rq[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(q_new[:-1, :-1, :-1], q_new_val)

    #
    # clipping on
    #
    wc = WaterConstituent(
        grid, True, backend=backend, backend_options=bo, storage_options=so
    )

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
