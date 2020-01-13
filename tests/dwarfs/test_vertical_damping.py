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
from pandas import Timedelta
import pytest

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD
from tasmania.python.utils.storage_utils import zeros

from tests.conf import backend as conf_backend, default_origin as conf_dorigin
from tests.utilities import (
    compare_arrays,
    st_domain,
    st_floats,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)


def assert_rayleigh(
    grid, depth, backend, default_origin, dt, phi_now, phi_new, phi_ref, phi_out
):
    dtype = phi_now.dtype
    ni, nj, nk = phi_now.shape

    vd = VD.factory(
        "rayleigh",
        grid,
        depth,
        0.01,
        time_units="s",
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=phi_now.shape,
    )

    rmat = vd._rmat

    vd(dt, phi_now, phi_new, phi_ref, phi_out)

    phi_val = phi_new[:ni, :nj, :nk] - dt.total_seconds() * rmat[:ni, :nj, :nk] * (
        phi_now[:ni, :nj, :nk] - phi_ref[:ni, :nj, :nk]
    )
    compare_arrays(phi_out[:, :, :depth], phi_val[:, :, :depth])
    compare_arrays(phi_out[:, :, depth:], phi_new[:ni, :nj, depth:nk])


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

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = cgrid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    shape = (cgrid.nx + dnx, cgrid.ny + dny, cgrid.nz + dnz)

    phi_now = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="phi_now",
    )
    phi_new = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="phi_new",
    )
    phi_ref = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="phi_ref",
    )

    dt = data.draw(
        st_timedeltas(min_value=Timedelta(seconds=0), max_value=Timedelta(hours=1)),
        label="dt",
    )

    depth = data.draw(hyp_st.integers(min_value=0, max_value=cgrid.nz), label="depth")

    # ========================================
    # test
    # ========================================
    phi_out = zeros(shape, backend, dtype, default_origin=default_origin)
    assert_rayleigh(
        cgrid, depth, backend, default_origin, dt, phi_now, phi_new, phi_ref, phi_out
    )


if __name__ == "__main__":
    pytest.main([__file__])
