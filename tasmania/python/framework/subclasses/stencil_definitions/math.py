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
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np

from gt4py import gtscript

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend=("numpy", "cupy"), stencil="abs")
def abs_numpy(in_field, out_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = np.abs(in_field[idx])


@stencil_definition.register(backend="gt4py*", stencil="abs")
def abs_gt4py(
    in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field if in_field > 0 else -in_field


@stencil_definition.register(backend=("numpy", "cupy"), stencil="add")
def add_numpy(in_a, in_b, out_c, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] + in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="add")
def add_gt4py(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a + in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="addsub")
def addsub_numpy(in_a, in_b, in_c, out_d, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_d[idx] = in_a[idx] + in_b[idx] - in_c[idx]


@stencil_definition.register(backend="gt4py*", stencil="addsub")
def addsub_gt4py(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    in_c: gtscript.Field["dtype"],
    out_d: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_d = in_a + in_b - in_c


@stencil_definition.register(backend="numpy", stencil="clip")
def clip_numpy(in_field, out_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = np.where(in_field[idx] > 0, in_field[idx], 0)


@stencil_definition.register(backend="cupy", stencil="clip")
def clip_cupy(in_field, out_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = cp.where(in_field[idx] > 0, in_field[idx], 0)


@stencil_definition.register(backend="gt4py*", stencil="clip")
def clip_gt4py(
    in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field if in_field > 0 else 0


@stencil_definition.register(backend=("numpy", "cupy"), stencil="fma")
def fma_numpy(in_a, in_b, out_c, *, f, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] + f * in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="fma")
def fma_gt4py(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
    *,
    f: "dtype"
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a + f * in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="iabs")
def iabs_numpy(inout_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = np.abs(inout_field[idx])


@stencil_definition.register(backend="gt4py*", stencil="iabs")
def iabs_gt4py(inout_field: gtscript.Field["dtype"]) -> None:
    with computation(PARALLEL), interval(...):
        inout_field = inout_field if inout_field > 0 else -inout_field


@stencil_definition.register(backend=("numpy", "cupy"), stencil="iadd")
def iadd_numpy(inout_a, in_b, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] += in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="iadd")
def iadd_gt4py(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="iaddsub")
def iaddsub_numpy(inout_a, in_b, in_c, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] += in_b[idx] - in_c[idx]


@stencil_definition.register(backend="gt4py*", stencil="iaddsub")
def iaddsub_gt4py(
    inout_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    in_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b - in_c


@stencil_definition.register(backend="numpy", stencil="iclip")
def iclip_numpy(inout_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = np.where(inout_field[idx] > 0, inout_field[idx], 0)


@stencil_definition.register(backend="cupy", stencil="iclip")
def iclip_cupy(inout_field, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = cp.where(inout_field[idx] > 0, inout_field[idx], 0)


@stencil_definition.register(backend="gt4py*", stencil="iclip")
def iclip_gt4py(inout_field: gtscript.Field["dtype"]) -> None:
    with computation(PARALLEL), interval(...):
        inout_field = inout_field if inout_field > 0 else 0


@stencil_definition.register(backend=("numpy", "cupy"), stencil="imul")
def imul_numpy(inout_a, in_b, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] *= in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="imul")
def imul_gt4py(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a * in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="iscale")
def iscale_numpy(inout_a, *, f, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] *= f


@stencil_definition.register(backend="gt4py*", stencil="iscale")
def iscale_gt4py(inout_a: gtscript.Field["dtype"], *, f: "dtype") -> None:
    with computation(PARALLEL), interval(...):
        inout_a = f * inout_a


@stencil_definition.register(backend=("numpy", "cupy"), stencil="isub")
def isub_numpy(inout_a, in_b, *, origin, domain, **kwargs) -> None:
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] -= in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="isub")
def isub_gt4py(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a - in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="mul")
def mul_numpy(in_a, in_b, out_c, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] * in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="mul")
def mul_gt4py(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a * in_b


@stencil_definition.register(backend=("numpy", "cupy"), stencil="scale")
def scale_numpy(in_a, out_a, *, f, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_a[idx] = f * in_a[idx]


@stencil_definition.register(backend="gt4py*", stencil="scale")
def scale_gt4py(
    in_a: gtscript.Field["dtype"],
    out_a: gtscript.Field["dtype"],
    *,
    f: "dtype"
) -> None:
    with computation(PARALLEL), interval(...):
        out_a = f * in_a


@stencil_definition.register(backend=("numpy", "cupy"), stencil="sub")
def sub_numpy(in_a, in_b, out_c, *, origin, domain, **kwargs):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] - in_b[idx]


@stencil_definition.register(backend="gt4py*", stencil="sub")
def sub_gt4py(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a - in_b
