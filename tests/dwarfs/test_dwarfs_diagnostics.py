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
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    aligned_index as conf_aligned_index,
    backend as conf_backend,
    dtype as conf_dtype,
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
    aligned_index = data.draw(
        st_one_of(conf_aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, cache=True, check_rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="r",
    )
    u = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="r",
    )
    v = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="r",
    )

    # ========================================
    # test bed
    # ========================================
    ru = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    rv = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv = HorizontalVelocity(
        grid, True, backend=backend, backend_options=bo, storage_options=so
    )

    hv.get_momenta(r, u, v, ru, rv)

    r_np, u_np, v_np = to_numpy(r), to_numpy(u), to_numpy(v)
    ru_val = (
        r_np[:-1, :-1, :-1] * 0.5 * (u_np[:-1, :-1, :-1] + u_np[1:, :-1, :-1])
    )
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = (
        r_np[:-1, :-1, :-1] * 0.5 * (v_np[:-1, :-1, :-1] + v_np[:-1, 1:, :-1])
    )
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    v_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    ru_np, rv_np = to_numpy(ru), to_numpy(rv)
    u_new_val = (ru_np[:-2, :] + ru_np[1:-1, :]) / (
        r_np[:-2, :] + r_np[1:-1, :]
    )
    compare_arrays(u_new[1:-1, :-1, :-1], u_new_val[:, :-1, :-1])
    v_new_val = (rv_np[:, :-2] + rv_np[:, 1:-1]) / (
        r_np[:, :-2] + r_np[:, 1:-1]
    )
    compare_arrays(v_new[:-1, 1:-1, :-1], v_new_val[:-1, :, :-1])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_horizontal_velocity(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf_aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, cache=True, check_rebuild=True)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="r",
    )
    u = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="r",
    )
    v = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="r",
    )

    # ========================================
    # test bed
    # ========================================
    ru = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    rv = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv = HorizontalVelocity(
        grid, False, backend=backend, backend_options=bo, storage_options=so
    )

    hv.get_momenta(r, u, v, ru, rv)

    r_np, u_np, v_np = to_numpy(r), to_numpy(u), to_numpy(v)
    ru_val = r_np[:-1, :-1, :-1] * u_np[:-1, :-1, :-1]
    compare_arrays(ru[:-1, :-1, :-1], ru_val)
    rv_val = r_np[:-1, :-1, :-1] * v_np[:-1, :-1, :-1]
    compare_arrays(rv[:-1, :-1, :-1], rv_val)

    u_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    v_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    hv.get_velocity_components(r, ru, rv, u_new, v_new)

    ru_np, rv_np = to_numpy(ru), to_numpy(rv)
    u_new_val = ru_np[:-1, :-1, :-1] / r_np[:-1, :-1, :-1]
    compare_arrays(u_new[:-1, :-1, :-1], u_new_val)
    v_new_val = rv_np[:-1, :-1, :-1] / r_np[:-1, :-1, :-1]
    compare_arrays(v_new[:-1, :-1, :-1], v_new_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_water_constituent(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf_aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, cache=True, check_rebuild=True)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=1,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="r",
    )
    q = data.draw(
        st_raw_field(
            shape=(nx + 1, ny + 1, nz + 1),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="q",
    )

    # ========================================
    # test bed
    # ========================================
    rq = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)
    q_new = zeros(backend, shape=(nx + 1, ny + 1, nz + 1), storage_options=so)

    #
    # clipping off
    #
    wc = WaterConstituent(
        grid, False, backend=backend, backend_options=bo, storage_options=so
    )

    wc.get_density_of_water_constituent(r, q, rq)
    r_np, q_np = to_numpy(r), to_numpy(q)
    rq_val = r_np[:-1, :-1, :-1] * q_np[:-1, :-1, :-1]
    compare_arrays(rq[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new)
    rq_np = to_numpy(rq)
    q_new_val = rq_np[:-1, :-1, :-1] / r_np[:-1, :-1, :-1]
    # compare_arrays(q_new[:-1, :-1, :-1], q_new_val)

    #
    # clipping on
    #
    wc = WaterConstituent(
        grid, True, backend=backend, backend_options=bo, storage_options=so
    )

    wc.get_density_of_water_constituent(r, q, rq)
    r_np, q_np = to_numpy(r), to_numpy(q)
    rq_val = r_np[:-1, :-1, :-1] * q_np[:-1, :-1, :-1]
    rq_val[rq_val < 0.0] = 0.0
    compare_arrays(rq[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new)
    rq_np = to_numpy(rq)
    q_new_val = rq_np[:-1, :-1, :-1] / r_np[:-1, :-1, :-1]
    q_new_val[q_new_val < 0.0] = 0.0
    compare_arrays(q_new[:-1, :-1, :-1], q_new_val)


if __name__ == "__main__":
    pytest.main([__file__])
