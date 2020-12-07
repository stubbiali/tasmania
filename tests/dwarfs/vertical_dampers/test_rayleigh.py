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
from pandas import Timedelta
import pytest

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_domain, st_one_of, st_raw_field, st_timedeltas
from tests.utilities import compare_arrays, hyp_settings


def assert_rayleigh(
    grid,
    depth,
    backend,
    backend_options,
    storage_options,
    dt,
    phi_now,
    phi_new,
    phi_ref,
    phi_out,
):
    ni, nj, nk = phi_now.shape

    vd = VD.factory(
        "rayleigh",
        grid,
        depth,
        0.01,
        time_units="s",
        backend=backend,
        backend_options=backend_options,
        storage_shape=phi_now.shape,
        storage_options=storage_options,
    )

    rmat = vd._rmat

    vd(dt, phi_now, phi_new, phi_ref, phi_out)

    phi_val = phi_new[:ni, :nj, :nk] - dt.total_seconds() * rmat[
        :ni, :nj, :nk
    ] * (phi_now[:ni, :nj, :nk] - phi_ref[:ni, :nj, :nk])
    compare_arrays(phi_out[:, :, :depth], phi_val[:, :, :depth])
    compare_arrays(phi_out[:, :, depth:], phi_val[:ni, :nj, depth:nk])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            backend=backend,
            dtype=dtype,
        ),
        label="grid",
    )
    cgrid = domain.numerical_grid

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
        st_timedeltas(
            min_value=Timedelta(seconds=0), max_value=Timedelta(hours=1)
        ),
        label="dt",
    )

    depth = data.draw(
        hyp_st.integers(min_value=0, max_value=cgrid.nz), label="depth"
    )

    # ========================================
    # test
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    phi_out = zeros(backend, shape=shape, storage_options=so)
    assert_rayleigh(
        cgrid, depth, backend, bo, so, dt, phi_now, phi_new, phi_ref, phi_out,
    )


if __name__ == "__main__":
    pytest.main([__file__])
