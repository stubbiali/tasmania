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

from tasmania.third_party import cupy as cp, gt4py, numba

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend="numpy", stencil="abs")
def abs_numpy(in_field, out_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = np.abs(in_field[idx])


@stencil_definition.register(backend="numpy", stencil="add")
def add_numpy(in_a, in_b, out_c, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] + in_b[idx]


@stencil_definition.register(backend="numpy", stencil="addsub")
def addsub_numpy(in_a, in_b, in_c, out_d, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_d[idx] = in_a[idx] + in_b[idx] - in_c[idx]


@stencil_definition.register(backend="numpy", stencil="clip")
def clip_numpy(in_field, out_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = np.where(in_field[idx] > 0, in_field[idx], 0)


@stencil_definition.register(backend="cupy", stencil="clip")
def clip_cupy(in_field, out_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = cp.where(in_field[idx] > 0, in_field[idx], 0)


@stencil_definition.register(backend="numpy", stencil="fma")
def fma_numpy(in_a, in_b, out_c, *, f, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] + f * in_b[idx]


@stencil_definition.register(backend="numpy", stencil="iabs")
def iabs_numpy(inout_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = np.abs(inout_field[idx])


@stencil_definition.register(backend="numpy", stencil="iadd")
def iadd_numpy(inout_a, in_b, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] += in_b[idx]


@stencil_definition.register(backend="numpy", stencil="iaddsub")
def iaddsub_numpy(inout_a, in_b, in_c, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] += in_b[idx] - in_c[idx]


@stencil_definition.register(backend="numpy", stencil="iclip")
def iclip_numpy(inout_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = np.where(inout_field[idx] > 0, inout_field[idx], 0)


@stencil_definition.register(backend="cupy", stencil="iclip")
def iclip_cupy(inout_field, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_field[idx] = cp.where(inout_field[idx] > 0, inout_field[idx], 0)


@stencil_definition.register(backend="numpy", stencil="imul")
def imul_numpy(inout_a, in_b, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] *= in_b[idx]


@stencil_definition.register(backend="numpy", stencil="iscale")
def iscale_numpy(inout_a, *, f, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] *= f


@stencil_definition.register(backend="numpy", stencil="isub")
def isub_numpy(inout_a, in_b, *, origin, domain) -> None:
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_a[idx] -= in_b[idx]


@stencil_definition.register(backend="numpy", stencil="mul")
def mul_numpy(in_a, in_b, out_c, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] * in_b[idx]


@stencil_definition.register(backend="numpy", stencil="scale")
def scale_numpy(in_a, out_a, *, f, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_a[idx] = f * in_a[idx]


@stencil_definition.register(backend="numpy", stencil="sub")
def sub_numpy(in_a, in_b, out_c, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_c[idx] = in_a[idx] - in_b[idx]


if True:  # cp:

    @stencil_definition.register(backend="cupy", stencil="abs")
    def abs_cupy(in_field, out_field, *, origin, domain):
        idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        out_field[idx] = cp.abs(in_field[idx])

    stencil_definition.register(add_numpy, "cupy", "add")

    stencil_definition.register(addsub_numpy, "cupy", "addsub")

    @stencil_definition.register(backend="cupy", stencil="clip")
    def clip_cupy(in_field, out_field, *, origin, domain):
        idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        out_field[idx] = cp.where(in_field[idx] > 0, in_field[idx], 0)

    stencil_definition.register(fma_numpy, "cupy", "fma")

    @stencil_definition.register(backend="cupy", stencil="iabs")
    def iabs_cupy(inout_field, *, origin, domain):
        idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        inout_field[idx] = cp.abs(inout_field[idx])

    stencil_definition.register(iadd_numpy, "cupy", "iadd")

    stencil_definition.register(iaddsub_numpy, "cupy", "iaddsub")

    @stencil_definition.register(backend="cupy", stencil="iclip")
    def iclip_cupy(inout_field, *, origin, domain):
        idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        inout_field[idx] = cp.where(inout_field[idx] > 0, inout_field[idx], 0)

    stencil_definition.register(imul_numpy, "cupy", "imul")

    stencil_definition.register(iscale_numpy, "cupy", "iscale")

    stencil_definition.register(isub_numpy, "cupy", "isub")

    stencil_definition.register(mul_numpy, "cupy", "mul")

    stencil_definition.register(scale_numpy, "cupy", "scale")

    stencil_definition.register(sub_numpy, "cupy", "sub")


if gt4py:
    from gt4py import gtscript

    @stencil_definition.register(backend="gt4py*", stencil="abs")
    def abs_gt4py(
        in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field if in_field > 0 else -in_field

    @stencil_definition.register(backend="gt4py*", stencil="add")
    def add_gt4py(
        in_a: gtscript.Field["dtype"],
        in_b: gtscript.Field["dtype"],
        out_c: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_c = in_a + in_b

    @stencil_definition.register(backend="gt4py*", stencil="addsub")
    def addsub_gt4py(
        in_a: gtscript.Field["dtype"],
        in_b: gtscript.Field["dtype"],
        in_c: gtscript.Field["dtype"],
        out_d: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_d = in_a + in_b - in_c

    @stencil_definition.register(backend="gt4py*", stencil="clip")
    def clip_gt4py(
        in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = in_field if in_field > 0 else 0

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

    @stencil_definition.register(backend="gt4py*", stencil="iabs")
    def iabs_gt4py(inout_field: gtscript.Field["dtype"]) -> None:
        with computation(PARALLEL), interval(...):
            inout_field = inout_field if inout_field > 0 else -inout_field

    @stencil_definition.register(backend="gt4py*", stencil="iadd")
    def iadd_gt4py(
        inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            inout_a = inout_a + in_b

    @stencil_definition.register(backend="gt4py*", stencil="iaddsub")
    def iaddsub_gt4py(
        inout_a: gtscript.Field["dtype"],
        in_b: gtscript.Field["dtype"],
        in_c: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            inout_a = inout_a + in_b - in_c

    @stencil_definition.register(backend="gt4py*", stencil="iclip")
    def iclip_gt4py(inout_field: gtscript.Field["dtype"]) -> None:
        with computation(PARALLEL), interval(...):
            inout_field = inout_field if inout_field > 0 else 0

    @stencil_definition.register(backend="gt4py*", stencil="imul")
    def imul_gt4py(
        inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            inout_a = inout_a * in_b

    @stencil_definition.register(backend="gt4py*", stencil="iscale")
    def iscale_gt4py(inout_a: gtscript.Field["dtype"], *, f: "dtype") -> None:
        with computation(PARALLEL), interval(...):
            inout_a = f * inout_a

    @stencil_definition.register(backend="gt4py*", stencil="isub")
    def isub_gt4py(
        inout_a: gtscript.Field["dtype"], in_b: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            inout_a = inout_a - in_b

    @stencil_definition.register(backend="gt4py*", stencil="mul")
    def mul_gt4py(
        in_a: gtscript.Field["dtype"],
        in_b: gtscript.Field["dtype"],
        out_c: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_c = in_a * in_b

    @stencil_definition.register(backend="gt4py*", stencil="scale")
    def scale_gt4py(
        in_a: gtscript.Field["dtype"],
        out_a: gtscript.Field["dtype"],
        *,
        f: "dtype"
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_a = f * in_a

    @stencil_definition.register(backend="gt4py*", stencil="sub")
    def sub_gt4py(
        in_a: gtscript.Field["dtype"],
        in_b: gtscript.Field["dtype"],
        out_c: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_c = in_a - in_b


if numba:

    @stencil_definition.register(backend="numba:cpu", stencil="abs")
    def abs_numba_cpu(in_field, out_field, *, origin, domain):
        def core_def(phi):
            return phi[0, 0, 0] if phi[0, 0, 0] > 0 else -phi[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(in_field[ib:ie, jb:je, kb:ke], out=out_field[ib:ie, jb:je, kb:ke])

    @stencil_definition.register(backend="numba:cpu", stencil="add")
    def add_numba_cpu(in_a, in_b, out_c, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] + b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=out_c[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="addsub")
    def addsub_numba_cpu(in_a, in_b, in_c, out_d, *, origin, domain):
        def core_def(a, b, c):
            return a[0, 0, 0] + b[0, 0, 0] - c[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            in_c[ib:ie, jb:je, kb:ke],
            out=out_d[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="clip")
    def clip_numba_cpu(in_field, out_field, *, origin, domain):
        def core_def(phi):
            return phi[0, 0, 0] if phi[0, 0, 0] > 0 else 0

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(in_field[ib:ie, jb:je, kb:ke], out=out_field[ib:ie, jb:je, kb:ke])

    @stencil_definition.register(backend="numba:cpu", stencil="fma")
    def fma_numba_cpu(in_a, in_b, out_c, *, f, origin, domain):
        def core_def(a, b, f):
            return a[0, 0, 0] + f * b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            f,
            out=out_c[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="fma")
    def fma_numba_cpu(in_a, in_b, out_c, *, f, origin, domain):
        def core_def(a, b, f):
            return a[0, 0, 0] + f * b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            f,
            out=out_c[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="iabs")
    def iabs_numba_cpu(inout_field, *, origin, domain):
        def core_def(phi):
            return phi[0, 0, 0] if phi[0, 0, 0] > 0 else -phi[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_field[ib:ie, jb:je, kb:ke],
            out=inout_field[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="iadd")
    def iadd_numba_cpu(inout_a, in_b, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] + b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=inout_a[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="iaddsub")
    def iaddsub_numba_cpu(inout_a, in_b, in_c, *, origin, domain):
        def core_def(a, b, c):
            return a[0, 0, 0] + b[0, 0, 0] - c[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            in_c[ib:ie, jb:je, kb:ke],
            out=inout_a[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="iclip")
    def iclip_numba_cpu(inout_field, *, origin, domain):
        def core_def(phi):
            return phi[0, 0, 0] if phi[0, 0, 0] > 0 else 0

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_field[ib:ie, jb:je, kb:ke],
            out=inout_field[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="imul")
    def imul_numba_cpu(inout_a, in_b, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] * b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=inout_a[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="iscale")
    def iscale_numba_cpu(inout_a, *, f, origin, domain):
        def core_def(a, f):
            return f * a[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(inout_a[ib:ie, jb:je, kb:ke], f, out=inout_a[ib:ie, jb:je, kb:ke])

    @stencil_definition.register(backend="numba:cpu", stencil="isub")
    def isub_numba_cpu(inout_a, in_b, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] - b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            inout_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=inout_a[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="mul")
    def mul_numba_cpu(in_a, in_b, out_c, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] * b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=out_c[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu", stencil="scale")
    def scale_numba_cpu(in_a, out_a, *, f, origin, domain):
        def core_def(a, f):
            return f * a[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(in_a[ib:ie, jb:je, kb:ke], f, out=out_a[ib:ie, jb:je, kb:ke])

    @stencil_definition.register(backend="numba:cpu", stencil="sub")
    def sub_numba_cpu(in_a, in_b, out_c, *, origin, domain):
        def core_def(a, b):
            return a[0, 0, 0] - b[0, 0, 0]

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_a[ib:ie, jb:je, kb:ke],
            in_b[ib:ie, jb:je, kb:ke],
            out=out_c[ib:ie, jb:je, kb:ke],
        )
