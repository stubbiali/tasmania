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

from tasmania.python.utils.storage import zeros

from tests.conf import default_origin as conf_dorigin
from tests.strategies import st_one_of, st_raw_field
from tests.utilities import compare_arrays


@gtscript.function
def add(a: gtscript.Field["undefined_dtype"], b: gtscript.Field["dtype"]):
    return a + b


def stencil_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
):
    from __externals__ import add

    with computation(PARALLEL), interval(...):
        out_c = add(in_a, in_b)


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
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = True
    backend = "numpy"
    dtype = np.float64
    default_origin = data.draw(st_one_of(conf_dorigin))

    nx = data.draw(hyp_st.integers(min_value=1, max_value=30))
    ny = data.draw(hyp_st.integers(min_value=1, max_value=30))
    nz = data.draw(hyp_st.integers(min_value=1, max_value=30))
    storage_shape = (nx, ny, nz)

    a = data.draw(
        st_raw_field(
            storage_shape,
            -1e5,
            1e5,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    b = data.draw(
        st_raw_field(
            storage_shape,
            -1e5,
            1e5,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )
    c = zeros(
        storage_shape,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    # ========================================
    # test bed
    # ========================================
    stencil = gtscript.stencil(
        definition=stencil_defs,
        backend=backend,
        rebuild=False,
        dtypes={"dtype": dtype},
        externals={"add": add},
    )

    stencil(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        storage_shape,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = a + b

    compare_arrays(c, c_val)


if __name__ == "__main__":
    pytest.main([__file__])
