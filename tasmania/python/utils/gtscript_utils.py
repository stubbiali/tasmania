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

from gt4py import gtscript


def set_annotations(func_handle, dtype):
    annotations = getattr(func_handle, "__annotations__", {})
    for arg in annotations:
        if isinstance(annotations[arg], gtscript._FieldDescriptor):
            annotations[arg] = gtscript.Field[dtype]
    return func_handle


@gtscript.function
def absolute(phi):
    return phi if phi > 0 else -phi


@gtscript.function
def positive(phi):
    return phi if phi > 0 else 0


@gtscript.function
def negative(phi):
    return -phi if phi < 0 else 0


def stencil_copy_defs(src: gtscript.Field[np.float64], dst: gtscript.Field[np.float64]):
    with computation(PARALLEL), interval(...):
        dst = src


def stencil_copychange_defs(
    src: gtscript.Field[np.float64], dst: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        dst = -src


def stencil_abs_defs(
    in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        out_field = in_field if in_field > 0 else -in_field


def stencil_iabs_defs(inout_field: gtscript.Field[np.float64]):
    with computation(PARALLEL), interval(...):
        inout_field = inout_field if inout_field > 0 else -inout_field


def stencil_add_defs(
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    out_c: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        out_c = in_a + in_b


def stencil_iadd_defs(
    inout_a: gtscript.Field[np.float64], in_b: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b


def stencil_sub_defs(
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    out_c: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        out_c = in_a - in_b


def stencil_isub_defs(
    inout_a: gtscript.Field[np.float64], in_b: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        inout_a = inout_a - in_b


def stencil_mul_defs(
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    out_c: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        out_c = in_a * in_b


def stencil_imul_defs(
    inout_a: gtscript.Field[np.float64], in_b: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        inout_a = inout_a * in_b


def stencil_scale_defs(
    in_a: gtscript.Field[np.float64], out_a: gtscript.Field[np.float64], *, f: float
):
    with computation(PARALLEL), interval(...):
        out_a = f * in_a


def stencil_iscale_defs(inout_a: gtscript.Field[np.float64], *, f: float):
    with computation(PARALLEL), interval(...):
        inout_a = f * inout_a


def stencil_addsub_defs(
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    in_c: gtscript.Field[np.float64],
    out_d: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        out_d = in_a + in_b - in_c


def stencil_iaddsub_defs(
    inout_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    in_c: gtscript.Field[np.float64],
):
    with computation(PARALLEL), interval(...):
        inout_a = inout_a + in_b - in_c


def stencil_fma_defs(
    in_a: gtscript.Field[np.float64],
    in_b: gtscript.Field[np.float64],
    out_c: gtscript.Field[np.float64],
    *,
    f: float
):
    with computation(PARALLEL), interval(...):
        out_c = in_a + f * in_b


def stencil_sts_rk2_0_defs(
    in_field: gtscript.Field[np.float64],
    in_field_prv: gtscript.Field[np.float64],
    in_tnd: gtscript.Field[np.float64],
    out_field: gtscript.Field[np.float64],
    *,
    dt: float
):
    with computation(PARALLEL), interval(...):
        out_field = 0.5 * (in_field + in_field_prv + dt * in_tnd)


def stencil_sts_rk3ws_0_defs(
    in_field: gtscript.Field[np.float64],
    in_field_prv: gtscript.Field[np.float64],
    in_tnd: gtscript.Field[np.float64],
    out_field: gtscript.Field[np.float64],
    *,
    dt: float
):
    with computation(PARALLEL), interval(...):
        out_field = (2.0 * in_field + in_field_prv + dt * in_tnd) / 3.0


def stencil_clip_defs(
    in_field: gtscript.Field[np.float64], out_field: gtscript.Field[np.float64]
):
    with computation(PARALLEL), interval(...):
        out_field = positive(in_field)


def stencil_iclip_defs(inout_field: gtscript.Field[np.float64]):
    with computation(PARALLEL), interval(...):
        inout_field = positive(inout_field)
