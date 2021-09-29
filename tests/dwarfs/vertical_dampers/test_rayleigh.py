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
from pandas import Timedelta
import pytest

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    aligned_index as conf_aligned_index,
    backend as conf_backend,
    dtype as conf_dtype,
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
    vd(dt, phi_now, phi_new, phi_ref, phi_out)

    rmat = to_numpy(vd._rmat)
    phi_now_np = to_numpy(phi_now)
    phi_new_np = to_numpy(phi_new)
    phi_ref_np = to_numpy(phi_ref)
    phi_val = phi_new_np[:ni, :nj, :nk] - dt.total_seconds() * rmat[
        :ni, :nj, :nk
    ] * (phi_now_np[:ni, :nj, :nk] - phi_ref_np[:ni, :nj, :nk])

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
    aligned_index = data.draw(
        st_one_of(conf_aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, cache=True, check_rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="grid",
    )
    ngrid = domain.numerical_grid

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    shape = (ngrid.nx + dnx, ngrid.ny + dny, ngrid.nz + dnz)

    phi_now = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            storage_options=so,
        ),
        label="phi_now",
    )
    phi_new = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            storage_options=so,
        ),
        label="phi_new",
    )
    phi_ref = data.draw(
        st_raw_field(
            shape,
            min_value=-1e10,
            max_value=1e10,
            backend=backend,
            storage_options=so,
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
        hyp_st.integers(min_value=0, max_value=ngrid.nz), label="depth"
    )

    # ========================================
    # test
    # ========================================
    phi_out = zeros(backend, shape=shape, storage_options=so)
    assert_rayleigh(
        ngrid,
        depth,
        backend,
        bo,
        so,
        dt,
        phi_now,
        phi_new,
        phi_ref,
        phi_out,
    )


if __name__ == "__main__":
    pytest.main([__file__])
