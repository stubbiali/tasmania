# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from tasmania.third_party import cupy, gt4py, numba

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend="numpy", stencil="irelax")
def irelax_numpy(in_gamma, in_phi_ref, inout_phi, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    inout_phi[idx] = np.where(
        in_gamma[idx] == 0.0,
        inout_phi[idx],
        np.where(
            in_gamma[idx] == 1.0,
            in_phi_ref[idx],
            inout_phi[idx]
            - in_gamma[idx] * (inout_phi[idx] - in_phi_ref[idx]),
        ),
    )


@stencil_definition.register(backend="numpy", stencil="relax")
def relax_numpy(in_gamma, in_phi, in_phi_ref, out_phi, *, origin, domain):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_phi[idx] = np.where(
        in_gamma[idx] == 0.0,
        in_phi[idx],
        np.where(
            in_gamma[idx] == 1.0,
            in_phi_ref[idx],
            in_phi[idx] - in_gamma[idx] * (in_phi[idx] - in_phi_ref[idx]),
        ),
    )


@stencil_definition.register(backend="numpy", stencil="sts_rk2_0")
def sts_rk2_0_numpy(
    in_field, in_field_prv, in_tnd, out_field, *, dt, origin, domain
):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = 0.5 * (
        in_field[idx] + in_field_prv[idx] + dt * in_tnd[idx]
    )


@stencil_definition.register(backend="numpy", stencil="sts_rk3ws_0")
def sts_rk3ws_0_numpy(
    in_field, in_field_prv, in_tnd, out_field, *, dt, origin, domain, **kwargs
):
    idx = tuple(slice(o, o + d) for o, d in zip(origin, domain))
    out_field[idx] = (
        2.0 * in_field[idx] + in_field_prv[idx] + dt * in_tnd[idx]
    ) / 3.0


if True:  # cupy:
    stencil_definition.register(irelax_numpy, "cupy", "irelax")
    stencil_definition.register(relax_numpy, "cupy", "relax")
    stencil_definition.register(sts_rk2_0_numpy, "cupy", "sts_rk2_0")
    stencil_definition.register(sts_rk3ws_0_numpy, "cupy", "sts_rk3ws_0")


if gt4py:
    from gt4py import gtscript

    @stencil_definition.register(backend="gt4py*", stencil="irelax")
    def irelax_gt4py(
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

    @stencil_definition.register(backend="gt4py*", stencil="relax")
    def relax_gt4py(
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

    @stencil_definition.register(backend="gt4py*", stencil="sts_rk2_0")
    def sts_rk2_0_gt4py(
        in_field: gtscript.Field["dtype"],
        in_field_prv: gtscript.Field["dtype"],
        in_tnd: gtscript.Field["dtype"],
        out_field: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = 0.5 * (in_field + in_field_prv + dt * in_tnd)

    @stencil_definition.register(backend="gt4py*", stencil="sts_rk3ws_0")
    def sts_rk3ws_0_gt4py(
        in_field: gtscript.Field["dtype"],
        in_field_prv: gtscript.Field["dtype"],
        in_tnd: gtscript.Field["dtype"],
        out_field: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_field = (2.0 * in_field + in_field_prv + dt * in_tnd) / 3.0


if numba:
    stencil_definition.register(irelax_numpy, "numba:cpu:numpy", "irelax")
    stencil_definition.register(relax_numpy, "numba:cpu:numpy", "relax")
    stencil_definition.register(
        sts_rk2_0_numpy, "numba:cpu:numpy", "sts_rk2_0"
    )
    stencil_definition.register(
        sts_rk3ws_0_numpy, "numba:cpu:numpy", "sts_rk3ws_0"
    )

    @stencil_definition.register(backend="numba:cpu:stencil", stencil="irelax")
    def irelax_numba_cpu(in_gamma, in_phi_ref, inout_phi, *, origin, domain):
        def core_def(gamma, phi, phi_ref):
            if gamma[0, 0, 0] == 0.0:
                return phi[0, 0, 0]
            elif gamma[0, 0, 0] == 1.0:
                return phi_ref[0, 0, 0]
            else:
                return phi[0, 0, 0] - gamma[0, 0, 0] * (
                    phi[0, 0, 0] - phi_ref[0, 0, 0]
                )

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_gamma[ib:ie, jb:je, kb:ke],
            inout_phi[ib:ie, jb:je, kb:ke],
            in_phi_ref[ib:ie, jb:je, kb:ke],
            out=inout_phi[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(backend="numba:cpu:stencil", stencil="relax")
    def relax_numba_cpu(
        in_gamma, in_phi, in_phi_ref, out_phi, *, origin, domain
    ):
        def core_def(gamma, phi, phi_ref):
            if gamma[0, 0, 0] == 0.0:
                return phi[0, 0, 0]
            elif gamma[0, 0, 0] == 1.0:
                return phi_ref[0, 0, 0]
            else:
                return phi[0, 0, 0] - gamma[0, 0, 0] * (
                    phi[0, 0, 0] - phi_ref[0, 0, 0]
                )

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_gamma[ib:ie, jb:je, kb:ke],
            in_phi[ib:ie, jb:je, kb:ke],
            in_phi_ref[ib:ie, jb:je, kb:ke],
            out=out_phi[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(
        backend="numba:cpu:stencil", stencil="sts_rk2_0"
    )
    def sts_rk2_0_numba_cpu(
        in_field, in_field_prv, in_tnd, out_field, *, dt, origin, domain
    ):
        def core_def(field, field_prv, field_tnd, dt):
            return 0.5 * (
                field[0, 0, 0] + field_prv[0, 0, 0] + dt * field_tnd[0, 0, 0]
            )

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_field[ib:ie, jb:je, kb:ke],
            in_field_prv[ib:ie, jb:je, kb:ke],
            in_tnd[ib:ie, jb:je, kb:ke],
            dt,
            out=out_field[ib:ie, jb:je, kb:ke],
        )

    @stencil_definition.register(
        backend="numba:cpu:stencil", stencil="sts_rk3ws_0"
    )
    def sts_rk3ws_0_numba_cpu(
        in_field, in_field_prv, in_tnd, out_field, *, dt, origin, domain
    ):
        def core_def(field, field_prv, field_tnd, dt):
            return (
                2.0 * field[0, 0, 0]
                + field_prv[0, 0, 0]
                + dt * field_tnd[0, 0, 0]
            ) / 3.0

        core = numba.stencil(core_def)

        ib, jb, kb = origin
        ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
        core(
            in_field[ib:ie, jb:je, kb:ke],
            in_field_prv[ib:ie, jb:je, kb:ke],
            in_tnd[ib:ie, jb:je, kb:ke],
            dt,
            out=out_field[ib:ie, jb:je, kb:ke],
        )
