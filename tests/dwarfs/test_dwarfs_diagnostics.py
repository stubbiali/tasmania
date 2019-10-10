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
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import compare_arrays, st_floats, st_one_of, st_physical_grid
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import compare_arrays, st_floats, st_one_of, st_physical_grid


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
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    assume(grid.nx > 1 and grid.ny > 1)
    dtype = grid.x.dtype

    phi = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(min_value=1, max_value=1e4),
            fill=hyp_st.nothing(),
        ),
        label="phi",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    r = phi[:-1, :-1, :]
    u = phi[1:, :-1, :]
    v = phi[:-1, 1:, :]

    field_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)
    halo = tuple(halo[i] if field_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(field_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)

    r_st = gt.storage.from_array(r, descriptor, backend=backend)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    ru_st = gt.storage.empty(descriptor, backend=backend)
    rv_st = gt.storage.empty(descriptor, backend=backend)

    hv = HorizontalVelocity(grid, True, backend=backend, rebuild=True)

    hv.get_momenta(r_st, u_st, v_st, ru_st, rv_st)

    ru_val = r[:-1, :-1, :-1] * 0.5 * (u[:-1, :-1, :-1] + u[1:, :-1, :-1])
    compare_arrays(ru_st.data[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * 0.5 * (v[:-1, :-1, :-1] + v[:-1, 1:, :-1])
    compare_arrays(rv_st.data[:-1, :-1, :-1], rv_val)

    u_new_st = gt.storage.empty(descriptor, backend=backend)
    v_new_st = gt.storage.empty(descriptor, backend=backend)

    hv.get_velocity_components(r_st, ru_st, rv_st, u_new_st, v_new_st)

    u_new_val = np.zeros_like(u, dtype=dtype)
    u_new_val[1:-1, :] = (ru_st.data[:-2, :] + ru_st.data[1:-1, :]) / (
        r[:-2, :] + r[1:-1, :]
    )
    compare_arrays(u_new_st.data[1:-1, :-1, :-1], u_new_val[1:-1, :-1, :-1])
    v_new_val = np.zeros_like(v, dtype=dtype)
    v_new_val[:, 1:-1] = (rv_st.data[:, :-2] + rv_st.data[:, 1:-1]) / (
        r[:, :-2] + r[:, 1:-1]
    )
    compare_arrays(v_new_st.data[:-1, 1:-1, :-1], v_new_val[:-1, 1:-1, :-1])


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
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    assume(grid.nx > 1 and grid.ny > 1)
    dtype = grid.x.dtype

    phi = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(min_value=1, max_value=1e4),
            fill=hyp_st.nothing(),
        ),
        label="phi",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    r = phi[:-1, :-1, :]
    u = phi[1:, :-1, :]
    v = phi[:-1, 1:, :]

    field_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)
    halo = tuple(halo[i] if field_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(field_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)

    r_st = gt.storage.from_array(r, descriptor, backend=backend)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    ru_st = gt.storage.empty(descriptor, backend=backend)
    rv_st = gt.storage.empty(descriptor, backend=backend)

    hv = HorizontalVelocity(grid, False, backend=backend, rebuild=True)

    hv.get_momenta(r_st, u_st, v_st, ru_st, rv_st)

    ru_val = r[:-1, :-1, :-1] * u[:-1, :-1, :-1]
    compare_arrays(ru_st.data[:-1, :-1, :-1], ru_val)
    rv_val = r[:-1, :-1, :-1] * v[:-1, :-1, :-1]
    compare_arrays(rv_st.data[:-1, :-1, :-1], rv_val)

    u_new_st = gt.storage.empty(descriptor, backend=backend)
    v_new_st = gt.storage.empty(descriptor, backend=backend)

    hv.get_velocity_components(r_st, ru_st, rv_st, u_new_st, v_new_st)

    u_new_val = ru_st.data[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(u_new_st.data[:-1, :-1, :-1], u_new_val)
    v_new_val = rv_st.data[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(v_new_st.data[:-1, :-1, :-1], v_new_val)


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
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    dtype = grid.x.dtype

    phi = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 1, grid.ny + 1, grid.nz + 2),
            elements=st_floats(min_value=-1e4, max_value=1e4),
            fill=hyp_st.nothing(),
        ),
        label="phi",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    r = phi[:, :, :-1]
    q = phi[:, :, 1:]

    field_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)
    halo = tuple(halo[i] if field_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(field_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)

    r_st = gt.storage.from_array(r, descriptor, backend=backend)
    q_st = gt.storage.from_array(q, descriptor, backend=backend)
    rq_st = gt.storage.empty(descriptor, backend=backend)
    q_new_st = gt.storage.empty(descriptor, backend=backend)

    #
    # clipping off
    #
    wc = WaterConstituent(grid, False, backend=backend, rebuild=True)

    wc.get_density_of_water_constituent(r_st, q_st, rq_st)
    rq_val = r[:-1, :-1, :-1] * q[:-1, :-1, :-1]
    compare_arrays(rq_st.data[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r_st, rq_st, q_new_st)
    q_new_val = rq_st.data[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    compare_arrays(q_new_st.data[:-1, :-1, :-1], q_new_val)

    #
    # clipping on
    #
    wc = WaterConstituent(grid, True, backend=backend, rebuild=True)

    wc.get_density_of_water_constituent(r_st, q_st, rq_st)
    rq_val = r[:-1, :-1, :-1] * q[:-1, :-1, :-1]
    rq_val[rq_val < 0.0] = 0.0
    compare_arrays(rq_st.data[:-1, :-1, :-1], rq_val)

    wc.get_mass_fraction_of_water_constituent_in_air(r_st, rq_st, q_new_st)
    q_new_val = rq_st.data[:-1, :-1, :-1] / r[:-1, :-1, :-1]
    q_new_val[q_new_val < 0.0] = 0.0
    compare_arrays(q_new_st.data[:-1, :-1, :-1], q_new_val)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_water_constituent()
