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
from gt4py.gtscript import PARALLEL, computation, interval

from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition

from tests import conf
from tests.strategies import st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


class Sum(StencilFactory):
    def __init__(self, backend, backend_options, storage_options):
        super().__init__(
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.sum = self.compile_stencil("sum")

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="sum")
    def sum_numpy(inout_a, in_b, *, origin, domain):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        inout_a[i, j, k] += in_b[i, j, k]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="sum")
    def sum_gt4py(
        inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
    ):
        with computation(PARALLEL), interval(...):
            inout_a += in_b[0, 0, 0]


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_sum(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(
        cache=True, check_rebuild=True, nopython=True, rebuild=False
    )
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    ni = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nj = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nk = data.draw(hyp_st.integers(min_value=1, max_value=30))
    shape = (ni, nj, nk)

    a = data.draw(
        st_raw_field(shape, -1e5, 1e5, backend=backend, storage_options=so)
    )
    b = data.draw(
        st_raw_field(shape, -1e5, 1e5, backend=backend, storage_options=so)
    )

    # ========================================
    # test bed
    # ========================================
    a_np = as_storage("numpy", data=a, storage_options=so)
    b_np = as_storage("numpy", data=b, storage_options=so)

    stencil_sum = Sum(backend, bo, so).sum
    stencil_sum(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(ni, nj, nk))

    a_np += b_np

    compare_arrays(a, a_np)


if __name__ == "__main__":
    pytest.main([__file__])
