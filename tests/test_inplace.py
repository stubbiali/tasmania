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
from copy import deepcopy
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

from gt4py import gtscript, storage as gt_storage

from tasmania.python.utils.storage_utils import zeros

try:
    from .conf import default_origin as conf_dorigin
    from .utils import compare_arrays, st_one_of, st_raw_field
except (ImportError, ModuleNotFoundError):
    from conf import default_origin as conf_dorigin
    from utils import compare_arrays, st_one_of, st_raw_field


def stencil_sum_defs(
    inout_a: gtscript.Field[np.float64], in_b: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        inout_a = inout_a[0, 0, 0] + in_b[0, 0, 0]


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_sum(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    ni = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nj = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nk = data.draw(hyp_st.integers(min_value=1, max_value=30))
    shape = (ni, nj, nk)

    backend = "numpy"
    dtype = np.float64
    default_origin = data.draw(st_one_of(conf_dorigin))

    a = data.draw(st_raw_field(shape, -1e5, 1e5, backend, dtype, default_origin))
    b = data.draw(st_raw_field(shape, -1e5, 1e5, backend, dtype, default_origin))

    # ========================================
    # test bed
    # ========================================
    decorator = gtscript.stencil(backend, rebuild=False)
    stencil_sum = decorator(stencil_sum_defs)

    a_dc = deepcopy(a)
    stencil_sum(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(ni, nj, nk))

    c = zeros(shape, backend, dtype, default_origin=default_origin)
    c[...] = a_dc + b

    compare_arrays(c, a)


def stencil_avg_defs(in_a: gtscript.Field[np.float64], out_a: gtscript.Field[np.float64]):
    from __externals__ import offi, offj

    with computation(PARALLEL), interval(...):
        out_a = 0.5 * (in_a[offi, offj, 0] + in_a[0, 0, 0])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_avg(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    ni = data.draw(hyp_st.integers(min_value=2, max_value=30))
    nj = data.draw(hyp_st.integers(min_value=2, max_value=30))
    nk = data.draw(hyp_st.integers(min_value=1, max_value=30))
    shape = (ni, nj, nk)

    offi = data.draw(hyp_st.integers(min_value=0, max_value=1))
    offj = data.draw(hyp_st.integers(min_value=0, max_value=1))

    backend = "numpy"
    dtype = np.float64
    default_origin = data.draw(st_one_of(conf_dorigin))

    a = data.draw(st_raw_field(shape, -1e5, 1e5, backend, dtype, default_origin))
    out_a = zeros((ni, nj, nk), backend, dtype, default_origin)

    # ========================================
    # test bed
    # ========================================
    decorator = gtscript.stencil(
        backend, externals={"offi": offi, "offj": offj}, rebuild=False
    )
    stencil_avg = decorator(stencil_avg_defs)

    stencil_avg(
        in_a=a, out_a=out_a, origin=(0, 0, 0), domain=(ni - offi, nj - offj, nk)
    )

    c = zeros(shape, backend, dtype, default_origin=default_origin)
    c[: ni - offi, : nj - offj] = 0.5 * (
        a[: ni - offi, : nj - offj] + a[offi:ni, offj:nj]
    )

    compare_arrays(c, out_a)


if __name__ == "__main__":
    pytest.main([__file__])
