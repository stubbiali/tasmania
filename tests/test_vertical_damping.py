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
from pandas import Timedelta
import pytest

import gridtools as gt
from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD

try:
    from .conf import backend as conf_backend, halo as conf_halo
    from .utils import compare_arrays, st_domain, st_floats, st_one_of, st_timedeltas
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo
    from utils import compare_arrays, st_domain, st_floats, st_one_of, st_timedeltas


def assert_rayleigh(
    grid, ni, nj, nk, depth, backend, halo, dt, phi_now, phi_new, phi_ref, phi_out
):
    dtype = phi_now.dtype
    shape = (ni, nj, nk)
    halo = tuple(halo[i] if shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    __phi_now = gt.storage.from_array(
        phi_now[:ni, :nj, :nk], descriptor, backend=backend
    )
    __phi_new = gt.storage.from_array(
        phi_new[:ni, :nj, :nk], descriptor, backend=backend
    )
    __phi_ref = gt.storage.from_array(
        phi_ref[:ni, :nj, :nk], descriptor, backend=backend
    )
    __phi_out = gt.storage.from_array(
        phi_out[:ni, :nj, :nk], descriptor, backend=backend
    )

    vd = VD.factory(
        "rayleigh",
        grid,
        (ni, nj, nk),
        depth,
        0.01,
        time_units="s",
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=True,
    )

    rmat = vd._rmat.data

    vd(dt, __phi_now, __phi_new, __phi_ref, __phi_out)

    phi_val = phi_new[:ni, :nj, :nk] - dt.total_seconds() * rmat[:ni, :nj, :nk] * (
        phi_now[:ni, :nj, :nk] - phi_ref[:ni, :nj, :nk]
    )
    compare_arrays(__phi_out.data[:, :, :depth], phi_val[:, :, :depth])
    compare_arrays(__phi_out.data[:, :, depth:], phi_new[:ni, :nj, depth:nk])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_rayleigh(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="grid",
    )

    cgrid = domain.numerical_grid

    phi = data.draw(
        st_arrays(
            cgrid.x.dtype,
            (cgrid.nx + 2, cgrid.ny + 2, cgrid.nz + 1),
            elements=st_floats(min_value=-1e10, max_value=1e10),
            fill=hyp_st.nothing(),
        ),
        label="phi",
    )

    dt = data.draw(
        st_timedeltas(min_value=Timedelta(seconds=0), max_value=Timedelta(hours=1)),
        label="dt",
    )

    depth = data.draw(hyp_st.integers(min_value=0, max_value=cgrid.nz), label="depth")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")

    # ========================================
    # test
    # ========================================
    ni = cgrid.nx + dnx
    nj = cgrid.ny + dny
    nk = cgrid.nz + dnz

    phi_now = phi[:ni, :nj, :nk]
    phi_new = phi[1 : ni + 1, :nj, :nk]
    phi_ref = phi[:ni, 1 : nj + 1, :nk]
    phi_out = np.zeros_like(phi_now, dtype=phi_now.dtype)

    assert_rayleigh(
        cgrid, ni, nj, nk, depth, backend, halo, dt, phi_now, phi_new, phi_ref, phi_out
    )


if __name__ == "__main__":
    pytest.main([__file__])
