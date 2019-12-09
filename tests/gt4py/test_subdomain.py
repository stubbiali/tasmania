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
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    out_c: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        out_c = in_a + in_b


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test(data):
    # gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(hyp_st.integers(min_value=1, max_value=30))
    ny = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nz = data.draw(hyp_st.integers(min_value=1, max_value=30))
    storage_shape = (nx, ny, nz)

    ni = data.draw(hyp_st.integers(min_value=1, max_value=nx))
    nj = data.draw(hyp_st.integers(min_value=1, max_value=ny))
    nk = data.draw(hyp_st.integers(min_value=1, max_value=nz))
    domain = (ni, nj, nk)

    oi = data.draw(hyp_st.integers(min_value=0, max_value=nx - ni))
    oj = data.draw(hyp_st.integers(min_value=0, max_value=ny - nj))
    ok = data.draw(hyp_st.integers(min_value=0, max_value=nz - nk))
    origin = (oi, oj, ok)

    backend = "numpy"
    dtype = np.float64
    default_origin = data.draw(st_one_of(conf_dorigin))

    a = data.draw(st_raw_field(storage_shape, -1e5, 1e5, backend, dtype, default_origin))
    b = data.draw(st_raw_field(storage_shape, -1e5, 1e5, backend, dtype, default_origin))
    c = zeros(storage_shape, backend, dtype, default_origin=default_origin)

    # ========================================
    # test bed
    # ========================================
    stencil_sum = gtscript.stencil(
        definition=stencil_sum_defs, backend=backend, rebuild=False
    )

    stencil_sum(in_a=a, in_b=b, out_c=c, origin=origin, domain=domain)

    i, j, k = slice(oi, oi + ni), slice(oj, oj + nj), slice(ok, ok + nk)
    c_val = zeros(storage_shape, backend, dtype, default_origin=default_origin)
    c_val[i, j, k] = a[i, j, k] + b[i, j, k]

    compare_arrays(c, c_val)


if __name__ == "__main__":
    pytest.main([__file__])
