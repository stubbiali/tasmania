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

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.isentropic.dynamics.horizontal_fluxes import IsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.implementations.minimal_horizontal_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import zeros

try:
    from .conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from .test_isentropic_horizontal_fluxes import (
        get_upwind_fluxes,
        get_centered_fluxes,
        get_maccormack_fluxes,
        get_third_order_upwind_fluxes,
        get_fifth_order_upwind_fluxes,
    )
    from .utils import st_domain, st_floats, st_one_of, st_raw_field, compare_arrays
except (ImportError, ModuleNotFoundError):
    from conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from test_isentropic_horizontal_fluxes import (
        get_upwind_fluxes,
        get_centered_fluxes,
        get_maccormack_fluxes,
        get_third_order_upwind_fluxes,
        get_fifth_order_upwind_fluxes,
    )
    from utils import st_domain, st_floats, st_one_of, st_raw_field, compare_arrays


class WrappingStencil:
    def __init__(self, core, nb, backend, dtype, default_origin, rebuild):
        self.core = core
        self.nb = nb
        self.backend = backend
        self.dtype = dtype
        self.default_origin = default_origin
        self.rebuild = rebuild

    def __call__(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        mi, mj, mk = s.shape

        stencil_args = {
            "s": s,
            "u": u,
            "v": v,
            "su": su,
            "sv": sv,
            "flux_s_x": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
            "flux_s_y": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
            "flux_su_x": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
            "flux_su_y": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
            "flux_sv_x": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
            "flux_sv_y": zeros((mi, mj, mk), self.backend, s.dtype, self.default_origin),
        }

        s_tnd_on = s_tnd is not None
        if s_tnd_on:
            stencil_args["s_tnd"] = s_tnd
        su_tnd_on = su_tnd is not None
        if su_tnd_on:
            stencil_args["su_tnd"] = su_tnd
        sv_tnd_on = sv_tnd is not None
        if sv_tnd_on:
            stencil_args["sv_tnd"] = sv_tnd

        moist = sqv is not None
        if moist:
            stencil_args["sqv"] = sqv
            stencil_args["flux_sqv_x"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )
            stencil_args["flux_sqv_y"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )
            stencil_args["sqc"] = sqc
            stencil_args["flux_sqc_x"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )
            stencil_args["flux_sqc_y"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )
            stencil_args["sqr"] = sqr
            stencil_args["flux_sqr_x"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )
            stencil_args["flux_sqr_y"] = zeros(
                (mi, mj, mk), self.backend, s.dtype, self.default_origin
            )

            if moist:
                qv_tnd_on = qv_tnd is not None
                if qv_tnd_on:
                    stencil_args["qv_tnd"] = qv_tnd
                qc_tnd_on = qc_tnd is not None
                if qc_tnd_on:
                    stencil_args["qc_tnd"] = qc_tnd
                qr_tnd_on = qr_tnd is not None
                if qv_tnd_on:
                    stencil_args["qr_tnd"] = qr_tnd

        # externals = self.core.externals.copy()
        externals = {
                "core": self.core.__call__,
                "moist": moist,
                "s_tnd_on": s_tnd_on,
                "su_tnd_on": su_tnd_on,
                "sv_tnd_on": sv_tnd_on,
            }
        if moist:
            externals.update(
                {"qv_tnd_on": qv_tnd_on, "qc_tnd_on": qc_tnd_on, "qr_tnd_on": qr_tnd_on}
            )

        decorator = gtscript.stencil(
            self.backend, externals=externals, rebuild=self.rebuild
        )
        stencil = decorator(self.stencil_defs)

        nb = self.nb
        stencil(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb - 1, nb - 1, 0)},
            domain=(mi - 2 * nb, mj - 2 * nb, mk)
        )

        return_list_names = [
            "flux_s_x",
            "flux_s_y",
            "flux_su_x",
            "flux_su_y",
            "flux_sv_x",
            "flux_sv_y",
        ]
        if moist:
            return_list_names += [
                "flux_sqv_x",
                "flux_sqv_y",
                "flux_sqc_x",
                "flux_sqc_y",
                "flux_sqr_x",
                "flux_sqr_y",
            ]
        return_list = tuple(stencil_args[name] for name in return_list_names)

        return return_list

    @staticmethod
    def stencil_defs(
        s: gtscript.Field[np.float64],
        u: gtscript.Field[np.float64],
        v: gtscript.Field[np.float64],
        su: gtscript.Field[np.float64],
        sv: gtscript.Field[np.float64],
        flux_s_x: gtscript.Field[np.float64],
        flux_s_y: gtscript.Field[np.float64],
        flux_su_x: gtscript.Field[np.float64],
        flux_su_y: gtscript.Field[np.float64],
        flux_sv_x: gtscript.Field[np.float64],
        flux_sv_y: gtscript.Field[np.float64],
        sqv: gtscript.Field[np.float64] = None,
        sqc: gtscript.Field[np.float64] = None,
        sqr: gtscript.Field[np.float64] = None,
        flux_sqv_x: gtscript.Field[np.float64] = None,
        flux_sqv_y: gtscript.Field[np.float64] = None,
        flux_sqc_x: gtscript.Field[np.float64] = None,
        flux_sqc_y: gtscript.Field[np.float64] = None,
        flux_sqr_x: gtscript.Field[np.float64] = None,
        flux_sqr_y: gtscript.Field[np.float64] = None,
        s_tnd: gtscript.Field[np.float64] = None,
        su_tnd: gtscript.Field[np.float64] = None,
        sv_tnd: gtscript.Field[np.float64] = None,
        qv_tnd: gtscript.Field[np.float64] = None,
        qc_tnd: gtscript.Field[np.float64] = None,
        qr_tnd: gtscript.Field[np.float64] = None,
        *,
        dt: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0
    ):
        from __externals__ import core, moist

        if not moist:
            with computation(PARALLEL), interval(...):
                flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = core(
                    dt=dt,
                    dx=dx,
                    dy=dy,
                    s=s,
                    u=u,
                    v=v,
                    su=su,
                    sv=sv,
                    s_tnd=s_tnd,
                    su_tnd=su_tnd,
                    sv_tnd=sv_tnd,
                )
        else:
            with computation(PARALLEL), interval(...):
                flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = core(
                    dt=dt,
                    dx=dx,
                    dy=dy,
                    s=s,
                    u=u,
                    v=v,
                    su=su,
                    sv=sv,
                    sqv=sqv,
                    sqc=sqc,
                    sqr=sqr,
                    s_tnd=s_tnd,
                    su_tnd=su_tnd,
                    sv_tnd=sv_tnd,
                    qv_tnd=qv_tnd,
                    qc_tnd=qc_tnd,
                    qr_tnd=qr_tnd,
                )


flux_properties = {
    "upwind": {"type": Upwind, "get_fluxes": get_upwind_fluxes},
    "centered": {"type": Centered, "get_fluxes": get_centered_fluxes},
    "third_order_upwind": {
        "type": ThirdOrderUpwind,
        "get_fluxes": get_third_order_upwind_fluxes,
    },
    "fifth_order_upwind": {
        "type": FifthOrderUpwind,
        "get_fluxes": get_fifth_order_upwind_fluxes,
    },
}


def validation(
    flux_scheme, domain, field, timestep, backend, dtype, default_origin, rebuild
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nb = domain.horizontal_boundary.nb
    flux_type = flux_properties[flux_scheme]["type"]
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    s = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    s[...] = field[: nx + 1, : ny + 1, :nz]
    u = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    u[...] = field[1 : nx + 2, : ny + 1, :nz]
    v = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    v[...] = field[: nx + 1, 1 : ny + 2, :nz]
    su = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    su[...] = field[1 : nx + 2, : ny + 1, :nz]
    sv = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    sv[...] = field[1 : nx + 2, 1 : ny + 2, :nz]
    sqv = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    sqv[...] = field[: nx + 1, : ny + 1, 1 : nz + 1]
    sqc = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    sqc[...] = field[1 : nx + 2, : ny + 1, 1 : nz + 1]
    sqr = zeros((nx + 1, ny + 1, nz), backend, dtype, default_origin)
    sqr[...] = field[1 : nx + 2, 1 : ny + 2, 1 : nz + 1]

    core = IsentropicMinimalHorizontalFlux.factory(flux_scheme)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, nb, backend, dtype, default_origin, rebuild=rebuild)

    #
    # dry
    #
    fsx, fsy, fsux, fsuy, fsvx, fsvy = ws(timestep, dx, dy, s, u, v, su, sv)

    flux_s_x, flux_s_y = get_fluxes(u, v, s)
    x = slice(nb - 1, grid.nx - nb)
    y = slice(nb - 1, grid.ny - nb)
    compare_arrays(fsx[x, y], flux_s_x[x, y])
    compare_arrays(fsy[x, y], flux_s_y[x, y])

    flux_su_x, flux_su_y = get_fluxes(u, v, su)
    compare_arrays(fsux[x, y], flux_su_x[x, y])
    compare_arrays(fsuy[x, y], flux_su_y[x, y])

    flux_sv_x, flux_sv_y = get_fluxes(u, v, sv)
    compare_arrays(fsvx[x, y], flux_sv_x[x, y])
    compare_arrays(fsvy[x, y], flux_sv_y[x, y])

    #
    # moist
    #
    fsx, fsy, fsux, fsuy, fsvx, fsvy, fsqvx, fsqvy, fsqcx, fsqcy, fsqrx, fsqry = ws(
        timestep, dx, dy, s, u, v, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
    )

    compare_arrays(fsx[x, y], flux_s_x[x, y])
    compare_arrays(fsy[x, y], flux_s_y[x, y])

    compare_arrays(fsux[x, y], flux_su_x[x, y])
    compare_arrays(fsuy[x, y], flux_su_y[x, y])

    compare_arrays(fsvx[x, y], flux_sv_x[x, y])
    compare_arrays(fsvy[x, y], flux_sv_y[x, y])

    flux_sqv_x, flux_sqv_y = get_fluxes(u, v, sqv)
    compare_arrays(fsqvx[x, y], flux_sqv_x[x, y])
    compare_arrays(fsqvy[x, y], flux_sqv_y[x, y])

    flux_sqc_x, flux_sqc_y = get_fluxes(u, v, sqc)
    compare_arrays(fsqcx[x, y], flux_sqc_x[x, y])
    compare_arrays(fsqcy[x, y], flux_sqc_y[x, y])

    flux_sqr_x, flux_sqr_y = get_fluxes(u, v, sqr)
    compare_arrays(fsqrx[x, y], flux_sqr_x[x, y])
    compare_arrays(fsqry[x, y], flux_sqr_y[x, y])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field((nx + 2, ny + 2, nz + 1), -1e4, 1e4, backend, dtype, default_origin),
        label="field",
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation(
        "upwind", domain, field, timestep, backend, dtype, default_origin, rebuild=False
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_centered(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field((nx + 2, ny + 2, nz + 1), -1e4, 1e4, backend, dtype, default_origin),
        label="field",
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation(
        "centered", domain, field, timestep, backend, dtype, default_origin, rebuild=False
    )


def _test_maccormack():
    ### TODO ###
    pass


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_third_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field((nx + 2, ny + 2, nz + 1), -1e4, 1e4, backend, dtype, default_origin),
        label="field",
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation(
        "third_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        default_origin,
        rebuild=False,
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_fifth_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    field = data.draw(
        st_raw_field((nx + 2, ny + 2, nz + 1), -1e4, 1e4, backend, dtype, default_origin),
        label="field",
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation(
        "fifth_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        default_origin,
        rebuild=False,
    )


if __name__ == "__main__":
    pytest.main([__file__])
