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
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

from gt4py import gtscript, storage as gt_storage

from tasmania.python.isentropic.dynamics.implementations.prognostic import (
    step_forward_euler_gt,
)
from tasmania.python.utils.gtscript_utils import (
    set_annotations,
    stencil_copy_defs,
    stencil_copychange_defs,
    absolute,
    positive,
    negative,
    stencil_abs_defs,
    stencil_iabs_defs,
    stencil_add_defs,
    stencil_iadd_defs,
    stencil_sub_defs,
    stencil_isub_defs,
    stencil_mul_defs,
    stencil_imul_defs,
    stencil_scale_defs,
    stencil_iscale_defs,
    stencil_addsub_defs,
    stencil_iaddsub_defs,
    stencil_fma_defs,
    stencil_sts_rk2_0_defs,
    stencil_sts_rk3ws_0_defs,
    stencil_clip_defs,
    stencil_iclip_defs,
    stencil_thomas_defs,
)
from tasmania.python.utils.storage_utils import zeros

from tests.conf import (
    backend as conf_backend,
    default_origin as conf_dorigin,
    datatype as conf_dtype,
)
from tests.strategies import st_floats, st_one_of, st_physical_grid, st_raw_field
from tests.utilities import compare_arrays


def assert_annotations(func_handle, dtype, *, subtests):
    args = {
        "s_now",
        "s_int",
        "s_new",
        "u_int",
        "v_int",
        "su_int",
        "sv_int",
        "mtg_int",
        "sqv_now",
        "sqv_int",
        "sqv_new",
        "sqc_now",
        "sqc_int",
        "sqc_new",
        "sqr_now",
        "sqr_int",
        "sqr_new",
        "s_tnd",
        "qv_tnd",
        "qc_tnd",
        "qr_tnd",
    }
    annotations = getattr(func_handle, "__annotations__", {})
    for arg in args:
        with subtests.test(arg=arg):
            assert arg in annotations
            assert isinstance(annotations[arg], gtscript._FieldDescriptor)
            assert annotations[arg].dtype == dtype


def test_set_annotations(subtests):
    # int
    set_annotations(step_forward_euler_gt, int)
    assert_annotations(step_forward_euler_gt, int, subtests=subtests)

    # float
    set_annotations(step_forward_euler_gt, float)
    assert_annotations(step_forward_euler_gt, float, subtests=subtests)

    # # np.float16
    # set_annotations(step_forward_euler_gt, np.float16)
    # assert_annotations(step_forward_euler_gt, np.float16)

    # np.float32
    set_annotations(step_forward_euler_gt, np.float32)
    assert_annotations(step_forward_euler_gt, np.float32, subtests=subtests)

    # np.float64
    set_annotations(step_forward_euler_gt, np.float64)
    assert_annotations(step_forward_euler_gt, np.float64, subtests=subtests)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_absolute(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    def stencil_absolute_defs(
        in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
    ):
        from __externals__ import absolute

        with computation(PARALLEL), interval(...):
            out_field = absolute(in_field)

    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_absolute = gtscript.stencil(
        backend=backend,
        definition=stencil_absolute_defs,
        dtypes={"dtype": dtype},
        externals={"absolute": absolute},
        rebuild=False,
    )

    stencil_absolute(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[...] = np.abs(src.data)

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_positive(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    def stencil_positive_defs(
        in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
    ):
        from __externals__ import positive

        with computation(PARALLEL), interval(...):
            out_field = positive(in_field)

    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_positive = gtscript.stencil(
        backend=backend,
        definition=stencil_positive_defs,
        dtypes={"dtype": dtype},
        externals={"positive": positive},
        rebuild=False,
    )

    stencil_positive(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[src > 0] = src[src > 0]

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_negative(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    def stencil_negative_defs(
        in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
    ):
        from __externals__ import negative

        with computation(PARALLEL), interval(...):
            out_field = negative(in_field)

    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_negative = gtscript.stencil(
        backend=backend,
        definition=stencil_negative_defs,
        dtypes={"dtype": dtype},
        externals={"negative": negative},
        rebuild=False,
    )

    stencil_negative(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[src < 0] = -src[src < 0]

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_copy(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    dst = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_copy = gtscript.stencil(
        backend=backend,
        definition=stencil_copy_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_copy(src=src, dst=dst, origin=(0, 0, 0), domain=(nx, ny, nz))

    compare_arrays(dst, src)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_copychange(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    dst = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_copychange = gtscript.stencil(
        backend=backend,
        definition=stencil_copychange_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_copychange(src=src, dst=dst, origin=(0, 0, 0), domain=(nx, ny, nz))

    compare_arrays(dst, -src)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_abs(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    set_annotations(stencil_abs_defs, dtype)
    stencil_abs = gtscript.stencil(
        backend=backend,
        definition=stencil_abs_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_abs(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[...] = np.abs(src.data)

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_iabs(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    src_dc = deepcopy(src)

    set_annotations(stencil_iabs_defs, dtype)
    stencil_iabs = gtscript.stencil(
        backend=backend,
        definition=stencil_iabs_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_iabs(inout_field=src, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[...] = np.abs(src_dc.data)

    compare_arrays(src, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_add(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    set_annotations(stencil_add_defs, dtype)
    stencil_add = gtscript.stencil(
        backend=backend,
        definition=stencil_add_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_add(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = a[...] + b[...]

    compare_arrays(c, c_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_iadd(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    a_dp = deepcopy(a)

    stencil_iadd = gtscript.stencil(
        backend=backend,
        definition=stencil_iadd_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_iadd(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(nx, ny, nz))

    a_dp += b

    compare_arrays(a, a_dp)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_sub(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_sub = gtscript.stencil(
        backend=backend,
        definition=stencil_sub_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_sub(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = a[...] - b[...]

    compare_arrays(c, c_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_isub(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    a_dp = deepcopy(a)

    stencil_isub = gtscript.stencil(
        backend=backend,
        definition=stencil_isub_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_isub(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(nx, ny, nz))

    a_dp -= b

    compare_arrays(a, a_dp)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_mul(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_mul = gtscript.stencil(
        backend=backend,
        definition=stencil_mul_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_mul(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = a[...] * b[...]

    compare_arrays(c, c_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_imul(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    a_dp = deepcopy(a)

    stencil_imul = gtscript.stencil(
        backend=backend,
        definition=stencil_imul_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_imul(inout_a=a, in_b=b, origin=(0, 0, 0), domain=(nx, ny, nz))

    a_dp *= b

    compare_arrays(a, a_dp)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_scale(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    # ========================================
    # test bed
    # ========================================
    c = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_scale = gtscript.stencil(
        backend=backend,
        definition=stencil_scale_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_scale(in_a=a, out_a=c, f=f, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = f * a[...]

    compare_arrays(c, c_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_iscale(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    # ========================================
    # test bed
    # ========================================
    a_dp = deepcopy(a)

    stencil_iscale = gtscript.stencil(
        backend=backend,
        definition=stencil_iscale_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_iscale(inout_a=a, f=f, origin=(0, 0, 0), domain=(nx, ny, nz))

    a_dp *= f

    compare_arrays(a, a_dp)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_addsub(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )
    c = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    d = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_addsub = gtscript.stencil(
        backend=backend,
        definition=stencil_addsub_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_addsub(
        in_a=a, in_b=b, in_c=c, out_d=d, origin=(0, 0, 0), domain=(nx, ny, nz)
    )

    d_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    d_val[...] = a[...] + b[...] - c[...]

    compare_arrays(d, d_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_iaddsub(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )
    c = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    a_dc = deepcopy(a)

    stencil_iaddsub = gtscript.stencil(
        backend=backend,
        definition=stencil_iaddsub_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_iaddsub(inout_a=a, in_b=b, in_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    a_dc[...] += b[...] - c[...]

    compare_arrays(a, a_dc)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_fma(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )
    f = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="f")

    # ========================================
    # test bed
    # ========================================
    c = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_fma = gtscript.stencil(
        backend=backend,
        definition=stencil_fma_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_fma(in_a=a, in_b=b, out_c=c, f=f, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c_val[...] = a[...] + f * b[...]

    compare_arrays(c, c_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_sts_rk2_0(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    field_prv = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field_prv",
    )
    tnd = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="tnd",
    )
    dt = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="dt")

    # ========================================
    # test bed
    # ========================================
    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_sts_rk2_0 = gtscript.stencil(
        backend=backend,
        definition=stencil_sts_rk2_0_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_sts_rk2_0(
        in_field=field,
        in_field_prv=field_prv,
        in_tnd=tnd,
        out_field=out,
        dt=dt,
        origin=(0, 0, 0),
        domain=(nx, ny, nz),
    )

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[...] = 0.5 * (field[...] + field_prv[...] + dt * tnd[...])

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_sts_rk3ws_0(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    field_prv = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field_prv",
    )
    tnd = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="tnd",
    )
    dt = data.draw(st_floats(min_value=-1e8, max_value=1e8), label="dt")

    # ========================================
    # test bed
    # ========================================
    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_sts_rk3ws_0 = gtscript.stencil(
        backend=backend,
        definition=stencil_sts_rk3ws_0_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_sts_rk3ws_0(
        in_field=field,
        in_field_prv=field_prv,
        in_tnd=tnd,
        out_field=out,
        dt=dt,
        origin=(0, 0, 0),
        domain=(nx, ny, nz),
    )

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[...] = (2.0 * field[...] + field_prv[...] + dt * tnd[...]) / 3.0

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_clip(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    out = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_clip = gtscript.stencil(
        backend=backend,
        definition=stencil_clip_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_clip(in_field=field, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[field > 0] = field[field > 0]

    compare_arrays(out, out_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_iclip(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    field_dc = deepcopy(field)

    stencil_iclip = gtscript.stencil(
        backend=backend,
        definition=stencil_iclip_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_iclip(inout_field=field, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out_val[field_dc > 0] = field_dc[field_dc > 0]

    compare_arrays(field, out_val)


def thomas_validation(a, b, c, d, x=None):
    nx, ny, nz = a.shape

    w = deepcopy(b)
    beta = deepcopy(b)
    delta = deepcopy(d)
    for i in range(nx):
        for j in range(ny):
            w[i, j, 0] = 0.0
            for k in range(1, nz):
                w[i, j, k] = (
                    a[i, j, k] / beta[i, j, k - 1]
                    if beta[i, j, k - 1] != 0.0
                    else a[i, j, k]
                )
                beta[i, j, k] = b[i, j, k] - w[i, j, k] * c[i, j, k - 1]
                delta[i, j, k] = d[i, j, k] - w[i, j, k] * delta[i, j, k - 1]

    x = deepcopy(b) if x is None else x
    for i in range(nx):
        for j in range(ny):
            x[i, j, -1] = (
                delta[i, j, -1] / beta[i, j, -1]
                if beta[i, j, -1] != 0.0
                else delta[i, j, -1] / b[i, j, -1]
            )
            for k in range(nz - 2, -1, -1):
                x[i, j, k] = (
                    (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1]) / beta[i, j, k]
                    if beta[i, j, k] != 0.0
                    else (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1]) / b[i, j, k]
                )

    return x


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_thomas_gt(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="dorigin")

    nx = data.draw(hyp_st.integers(min_value=1, max_value=30), label="nx")
    ny = data.draw(hyp_st.integers(min_value=1, max_value=30), label="ny")
    nz = data.draw(hyp_st.integers(min_value=2, max_value=30), label="nz")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="a",
    )
    b = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )
    b[b == 0.0] = 1.0
    c = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="c",
    )
    d = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="d",
    )

    # ========================================
    # test bed
    # ========================================
    x = zeros(
        (nx, ny, nz),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    stencil_thomas = gtscript.stencil(
        backend=backend,
        definition=stencil_thomas_defs,
        dtypes={"dtype": dtype},
        rebuild=False,
    )

    stencil_thomas(a=a, b=b, c=c, d=d, x=x, origin=(0, 0, 0), domain=(nx, ny, nz))

    x_val = thomas_validation(a, b, c, d)

    compare_arrays(x, x_val)

    try:
        i = data.draw(hyp_st.integers(min_value=0, max_value=nx - 1))
        j = data.draw(hyp_st.integers(min_value=0, max_value=ny - 1))
        m = np.diag(a[i, j, 1:], -1) + np.diag(b[i, j, :]) + np.diag(c[i, j, :-1], 1)
        d_val = zeros(
            (1, 1, nz),
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        for k in range(nz):
            for p in range(nz):
                d_val[0, 0, k] += m[k, p] * x_val[i, j, p]

        compare_arrays(d_val, d[i, j])
    except AssertionError:
        print("Numerical verification of Thomas' algorithm failed.")


if __name__ == "__main__":
    pytest.main([__file__])
