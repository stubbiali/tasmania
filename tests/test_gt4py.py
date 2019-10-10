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

import gridtools as gt
from tasmania.python.utils.storage_utils import zeros

try:
    from .conf import halo as conf_halo
    from .utils import compare_arrays, st_one_of, st_raw_field
except (ImportError, ModuleNotFoundError):
    from conf import halo as conf_halo
    from utils import compare_arrays, st_one_of, st_raw_field


def stencil_sum_defs(inout_a: gt.storage.f64_sd, in_b: gt.storage.f64_sd):
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
    # ========================================
    # random data generation
    # ========================================
    ni = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nj = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nk = data.draw(hyp_st.integers(min_value=1, max_value=30))
    shape = (ni, nj, nk)

    backend = "gtmc"
    dtype = np.float64
    halo = data.draw(st_one_of(conf_halo))

    a = data.draw(st_raw_field(shape, -1e5, 1e5, backend, dtype, halo))
    b = data.draw(st_raw_field(shape, -1e5, 1e5, backend, dtype, halo))

    # ========================================
    # test bed
    # ========================================
    decorator = gt.stencil(backend, rebuild=False)
    stencil_sum = decorator(stencil_sum_defs)

    a_dc = deepcopy(a)
    stencil_sum(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(ni, nj, nk))

    c = zeros(shape, backend, dtype, halo=halo)
    c[...] = a_dc + b

    compare_arrays(c, a)


if __name__ == "__main__":
    pytest.main([__file__])
