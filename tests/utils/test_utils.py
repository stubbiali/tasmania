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
import numpy as np
import pytest

from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import feed_module, thomas_numpy

from tests.conf import datatype as conf_dtype
from tests.utilities import compare_arrays, st_one_of, st_raw_field
from tests.utils.test_gtscript_utils import thomas_validation


def test_feed_module():
    from tests.utils import namelist
    from tests.utils import namelist_baseline as baseline

    feed_module(target=namelist, source=baseline)

    assert hasattr(namelist, "bar")
    assert namelist.bar == 1.0
    assert hasattr(namelist, "foo")
    assert namelist.foo is True
    assert hasattr(namelist, "pippo")
    assert namelist.pippo == "Hello, world!"
    assert hasattr(namelist, "franco")
    assert namelist.franco == "Hello, world!"
    assert hasattr(namelist, "ciccio")
    assert namelist.ciccio == np.float64


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_thomas_numpy(data):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(hyp_st.integers(min_value=1, max_value=30), label="nx")
    ny = data.draw(hyp_st.integers(min_value=1, max_value=30), label="ny")
    nz = data.draw(hyp_st.integers(min_value=2, max_value=30), label="nz")

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=False,
            dtype=dtype,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=False,
            dtype=dtype,
        ),
        label="b",
    )
    b[b == 0.0] = 1.0
    c = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=False,
            dtype=dtype,
        ),
        label="c",
    )
    d = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=False,
            dtype=dtype,
        ),
        label="d",
    )

    # ========================================
    # test bed
    # ========================================
    x = zeros((nx, ny, nz), gt_powered=False, dtype=dtype)

    thomas_numpy(
        a=a, b=b, c=c, d=d, out=x, i=slice(0, nx), j=slice(0, ny), kstart=0, kstop=nz
    )

    x_val = thomas_validation(a, b, c, d)

    compare_arrays(x, x_val)

    try:
        i = data.draw(hyp_st.integers(min_value=0, max_value=nx - 1))
        j = data.draw(hyp_st.integers(min_value=0, max_value=ny - 1))
        m = np.diag(a[i, j, 1:], -1) + np.diag(b[i, j, :]) + np.diag(c[i, j, :-1], 1)
        d_val = zeros((1, 1, nz), gt_powered=False, dtype=dtype)
        for k in range(nz):
            for p in range(nz):
                d_val[0, 0, k] += m[k, p] * x_val[i, j, p]

        compare_arrays(d_val, d[i, j])
    except AssertionError:
        print("Numerical verification of Thomas' algorithm failed.")


if __name__ == "__main__":
    pytest.main([__file__])
