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
from typing import Callable

from gt4py import gtscript
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

from tasmania.python.utils import typingx


def set_annotations(func_handle: Callable, dtype: typingx.dtype_t) -> Callable:
    annotations = getattr(func_handle, "__annotations__", {})
    for arg in annotations:
        if isinstance(annotations[arg], gtscript._FieldDescriptor):
            annotations[arg] = gtscript.Field[dtype]
    return func_handle


@gtscript.function
def absolute(phi: typingx.GTField) -> typingx.GTField:
    return phi if phi > 0 else -phi


@gtscript.function
def positive(phi: typingx.GTField) -> typingx.GTField:
    return phi if phi > 0 else 0


@gtscript.function
def negative(phi: typingx.GTField) -> typingx.GTField:
    return -phi if phi < 0 else 0


def stencil_copy_defs(
    src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        dst = src


def stencil_copychange_defs(
    src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        dst = -src


def stencil_abs_defs(
    in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = in_field if in_field > 0 else -in_field


def stencil_iabs_defs(inout_field: gtscript.Field["dtype"]) -> None:
    with computation(PARALLEL), interval(...):
        inout_field = inout_field if inout_field > 0 else -inout_field


def stencil_add_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a + in_b


def stencil_iadd_defs(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b


def stencil_sub_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a - in_b


def stencil_isub_defs(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a - in_b


def stencil_mul_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a * in_b


def stencil_imul_defs(
    inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a * in_b


def stencil_scale_defs(
    in_a: gtscript.Field["dtype"], out_a: gtscript.Field["dtype"], *, f: float
) -> None:
    with computation(PARALLEL), interval(...):
        out_a = f * in_a


def stencil_iscale_defs(inout_a: gtscript.Field["dtype"], *, f: float) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = f * inout_a


def stencil_addsub_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    in_c: gtscript.Field["dtype"],
    out_d: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        out_d = in_a + in_b - in_c


def stencil_iaddsub_defs(
    inout_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    in_c: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b - in_c


def stencil_fma_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
    *,
    f: float
) -> None:
    with computation(PARALLEL), interval(...):
        out_c = in_a + f * in_b


def stencil_sts_rk2_0_defs(
    in_field: gtscript.Field["dtype"],
    in_field_prv: gtscript.Field["dtype"],
    in_tnd: gtscript.Field["dtype"],
    out_field: gtscript.Field["dtype"],
    *,
    dt: float
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = 0.5 * (in_field + in_field_prv + dt * in_tnd)


def stencil_sts_rk3ws_0_defs(
    in_field: gtscript.Field["dtype"],
    in_field_prv: gtscript.Field["dtype"],
    in_tnd: gtscript.Field["dtype"],
    out_field: gtscript.Field["dtype"],
    *,
    dt: float
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = (2.0 * in_field + in_field_prv + dt * in_tnd) / 3.0


def stencil_clip_defs(
    in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        out_field = positive(in_field)


def stencil_iclip_defs(inout_field: gtscript.Field["dtype"]) -> None:
    with computation(PARALLEL), interval(...):
        inout_field = positive(inout_field)


def stencil_thomas_defs(
    a: gtscript.Field["dtype"],
    b: gtscript.Field["dtype"],
    c: gtscript.Field["dtype"],
    d: gtscript.Field["dtype"],
    x: gtscript.Field["dtype"],
) -> None:
    # """ The Thomas' algorithm to solve a tridiagonal system of equations. """
    with computation(FORWARD), interval(0, 1):
        w = 0.0
        beta = b[0, 0, 0]
        delta = d[0, 0, 0]
    with computation(FORWARD), interval(1, None):
        w = (
            a[0, 0, 0] / beta[0, 0, -1]
            if beta[0, 0, -1] != 0.0
            else a[0, 0, 0]
        )
        beta = b[0, 0, 0] - w[0, 0, 0] * c[0, 0, -1]
        delta = d[0, 0, 0] - w[0, 0, 0] * delta[0, 0, -1]

    with computation(BACKWARD), interval(-1, None):
        x = (
            delta[0, 0, 0] / beta[0, 0, 0]
            if beta[0, 0, 0] != 0.0
            else delta[0, 0, 0] / b[0, 0, 0]
        )
    with computation(BACKWARD), interval(0, -1):
        x = (
            (delta[0, 0, 0] - c[0, 0, 0] * x[0, 0, 1]) / beta[0, 0, 0]
            if beta[0, 0, 0] != 0.0
            else (delta[0, 0, 0] - c[0, 0, 0] * x[0, 0, 1]) / b[0, 0, 0]
        )


def stencil_relax_defs(
    in_gamma: gtscript.Field["dtype"],
    in_phi: gtscript.Field["dtype"],
    in_phi_ref: gtscript.Field["dtype"],
    out_phi: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        if in_gamma == 0.0:
            out_phi = in_phi
        elif in_gamma == 1.0:
            out_phi = in_phi_ref
        else:
            out_phi = in_phi - in_gamma * (in_phi - in_phi_ref)


def stencil_irelax_defs(
    in_gamma: gtscript.Field["dtype"],
    in_phi_ref: gtscript.Field["dtype"],
    inout_phi: gtscript.Field["dtype"],
) -> None:
    with computation(PARALLEL), interval(...):
        if in_gamma == 0.0:
            inout_phi = inout_phi
        elif in_gamma == 1.0:
            inout_phi = in_phi_ref
        else:
            inout_phi = inout_phi - in_gamma * (inout_phi - in_phi_ref)
