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

from gt4py import gtscript

from tasmania.python.isentropic.dynamics.implementations.prognostic import (
    step_forward_euler,
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
)
from tasmania.python.utils.storage_utils import zeros

try:
    from .conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from .utils import (
        compare_arrays,
        st_floats,
        st_one_of,
        st_physical_grid,
        st_raw_field,
    )
except (ImportError, ModuleNotFoundError):
    from conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from utils import compare_arrays, st_floats, st_one_of, st_physical_grid, st_raw_field


def assert_annotations(func_handle, dtype):
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
        assert arg in annotations
        assert isinstance(annotations[arg], gtscript._FieldDescriptor)
        assert annotations[arg].dtype == dtype


def test_set_annotations():
    # int
    set_annotations(step_forward_euler, int)
    assert_annotations(step_forward_euler, int)

    # float
    set_annotations(step_forward_euler, float)
    assert_annotations(step_forward_euler, float)

    # np.float16
    set_annotations(step_forward_euler, np.float16)
    assert_annotations(step_forward_euler, np.float16)

    # np.float32
    set_annotations(step_forward_euler, np.float32)
    assert_annotations(step_forward_euler, np.float32)

    # np.float64
    set_annotations(step_forward_euler, np.float64)
    assert_annotations(step_forward_euler, np.float64)


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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
        in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
    ):
        from __externals__ import absolute

        with computation(PARALLEL), interval(...):
            out_field = absolute(in_field)

    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_absolute_defs, dtype)
    stencil_absolute = gtscript.stencil(
        backend=backend,
        definition=stencil_absolute_defs,
        externals={"absolute": absolute},
        rebuild=False,
    )

    stencil_absolute(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
        in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
    ):
        from __externals__ import positive

        with computation(PARALLEL), interval(...):
            out_field = positive(in_field)

    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_positive_defs, dtype)
    stencil_positive = gtscript.stencil(
        backend=backend,
        definition=stencil_positive_defs,
        externals={"positive": positive},
        rebuild=False,
    )

    stencil_positive(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
        in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
    ):
        from __externals__ import negative

        with computation(PARALLEL), interval(...):
            out_field = negative(in_field)

    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_negative_defs, dtype)
    stencil_negative = gtscript.stencil(
        backend=backend,
        definition=stencil_negative_defs,
        externals={"negative": negative},
        rebuild=False,
    )

    stencil_negative(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    dst = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_copy_defs, dtype)
    stencil_copy = gtscript.stencil(
        backend=backend, definition=stencil_copy_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    dst = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_copychange_defs, dtype)
    stencil_copychange = gtscript.stencil(
        backend=backend, definition=stencil_copychange_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="src",
    )

    # ========================================
    # test bed
    # ========================================
    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_abs_defs, dtype)
    stencil_abs = gtscript.stencil(
        backend=backend, definition=stencil_abs_defs, rebuild=False
    )

    stencil_abs(in_field=src, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    src = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
        backend=backend, definition=stencil_iabs_defs, rebuild=False
    )

    stencil_iabs(inout_field=src, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_add_defs, dtype)
    stencil_add = gtscript.stencil(
        backend=backend, definition=stencil_add_defs, rebuild=False
    )

    stencil_add(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_iadd_defs, dtype)
    stencil_iadd = gtscript.stencil(
        backend=backend, definition=stencil_iadd_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_sub_defs, dtype)
    stencil_sub = gtscript.stencil(
        backend=backend, definition=stencil_sub_defs, rebuild=False
    )

    stencil_sub(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_isub_defs, dtype)
    stencil_isub = gtscript.stencil(
        backend=backend, definition=stencil_isub_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    c = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_mul_defs, dtype)
    stencil_mul = gtscript.stencil(
        backend=backend, definition=stencil_mul_defs, rebuild=False
    )

    stencil_mul(in_a=a, in_b=b, out_c=c, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_imul_defs, dtype)
    stencil_imul = gtscript.stencil(
        backend=backend, definition=stencil_imul_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
    c = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_scale_defs, dtype)
    stencil_scale = gtscript.stencil(
        backend=backend, definition=stencil_scale_defs, rebuild=False
    )

    stencil_scale(in_a=a, out_a=c, f=f, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_iscale_defs, dtype)
    stencil_iscale = gtscript.stencil(
        backend=backend, definition=stencil_iscale_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="b",
    )

    # ========================================
    # test bed
    # ========================================
    d = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_addsub_defs, dtype)
    stencil_addsub = gtscript.stencil(
        backend=backend, definition=stencil_addsub_defs, rebuild=False
    )

    stencil_addsub(in_a=a, in_b=b, in_c=c, out_d=d, origin=(0, 0, 0), domain=(nx, ny, nz))

    d_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_iaddsub_defs, dtype)
    stencil_iaddsub = gtscript.stencil(
        backend=backend, definition=stencil_iaddsub_defs, rebuild=False
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    a = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
    c = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_fma_defs, dtype)
    stencil_fma = gtscript.stencil(
        backend=backend, definition=stencil_fma_defs, rebuild=False
    )

    stencil_fma(in_a=a, in_b=b, out_c=c, f=f, origin=(0, 0, 0), domain=(nx, ny, nz))

    c_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_sts_rk2_0_defs, dtype)
    stencil_sts_rk2_0 = gtscript.stencil(
        backend=backend, definition=stencil_sts_rk2_0_defs, rebuild=False
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

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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
    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_sts_rk3ws_0_defs, dtype)
    stencil_sts_rk3ws_0 = gtscript.stencil(
        backend=backend, definition=stencil_sts_rk3ws_0_defs, rebuild=False
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

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    # ========================================
    # test bed
    # ========================================
    out = zeros((nx, ny, nz), backend, dtype, default_origin)

    set_annotations(stencil_clip_defs, dtype)
    stencil_clip = gtscript.stencil(
        backend=backend, definition=stencil_clip_defs, rebuild=False
    )

    stencil_clip(in_field=field, out_field=out, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
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
    # comment the following line to prevent segfault
    # gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field(
            shape=(nx, ny, nz),
            min_value=-1e4,
            max_value=1e4,
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

    set_annotations(stencil_iclip_defs, dtype)
    stencil_iclip = gtscript.stencil(
        backend=backend, definition=stencil_iclip_defs, rebuild=False
    )

    stencil_iclip(inout_field=field, origin=(0, 0, 0), domain=(nx, ny, nz))

    out_val = zeros((nx, ny, nz), backend, dtype, default_origin)
    out_val[field_dc > 0] = field_dc[field_dc > 0]

    compare_arrays(field, out_val)


if __name__ == "__main__":
    pytest.main([__file__])
