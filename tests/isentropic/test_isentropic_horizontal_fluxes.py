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
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest

from gt4py import gtscript

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.subclasses.horizontal_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.utils import get_gt_backend, is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


def test_registry():
    assert "upwind" in IsentropicHorizontalFlux.registry
    assert IsentropicHorizontalFlux.registry["upwind"] == Upwind
    assert "centered" in IsentropicHorizontalFlux.registry
    assert IsentropicHorizontalFlux.registry["centered"] == Centered
    assert "third_order_upwind" in IsentropicHorizontalFlux.registry
    assert (
        IsentropicHorizontalFlux.registry["third_order_upwind"]
        == ThirdOrderUpwind
    )
    assert "fifth_order_upwind" in IsentropicHorizontalFlux.registry
    assert (
        IsentropicHorizontalFlux.registry["fifth_order_upwind"]
        == FifthOrderUpwind
    )


def test_factory():
    obj = IsentropicHorizontalFlux.factory("upwind", backend="numpy")
    assert isinstance(obj, Upwind)
    obj = IsentropicHorizontalFlux.factory("centered", backend="numpy")
    assert isinstance(obj, Centered)
    obj = IsentropicHorizontalFlux.factory(
        "third_order_upwind", backend="numpy"
    )
    assert isinstance(obj, ThirdOrderUpwind)
    obj = IsentropicHorizontalFlux.factory(
        "fifth_order_upwind", backend="numpy"
    )
    assert isinstance(obj, FifthOrderUpwind)


class WrappingStencil(StencilFactory):
    def __init__(
        self, cls, scheme, nb, backend, backend_options, storage_options
    ):
        super().__init__(backend, backend_options, storage_options)
        self.nb = nb
        self.core = cls.factory(scheme, backend=backend)

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}

        self.backend_options.externals = self.core.externals.copy()
        self.backend_options.externals[
            "get_flux_dry"
        ] = self.core.stencil_subroutine("flux_dry")
        self.backend_options.externals[
            "get_flux_moist"
        ] = self.core.stencil_subroutine("flux_moist")

        self.stencil_dry = self.compile("stencil_dry")
        self.stencil_moist = self.compile("stencil_moist")

    def call_dry(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
    ):
        mi, mj, mk = s.shape
        nb = self.nb

        stencil_args = {
            "s": s,
            "u": u,
            "v": v,
            "su": su,
            "sv": sv,
            "flux_s_x": self.zeros(shape=(mi, mj, mk)),
            "flux_s_y": self.zeros(shape=(mi, mj, mk)),
            "flux_su_x": self.zeros(shape=(mi, mj, mk)),
            "flux_su_y": self.zeros(shape=(mi, mj, mk)),
            "flux_sv_x": self.zeros(shape=(mi, mj, mk)),
            "flux_sv_y": self.zeros(shape=(mi, mj, mk)),
        }
        if mtg is not None:
            stencil_args["mtg"] = mtg

        if s_tnd is not None:
            stencil_args["s_tnd"] = s_tnd
        if su_tnd is not None:
            stencil_args["su_tnd"] = su_tnd
        if sv_tnd is not None:
            stencil_args["sv_tnd"] = sv_tnd

        self.stencil_dry(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
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
        return_list = tuple(stencil_args[name] for name in return_list_names)

        return return_list

    def call_moist(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        sqv,
        sqc,
        sqr,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        mi, mj, mk = sqv.shape
        nb = self.nb

        stencil_args = {
            "s": s,
            "u": u,
            "v": v,
            "sqv": sqv,
            "sqc": sqc,
            "sqr": sqr,
            "flux_sqv_x": self.zeros(shape=(mi, mj, mk)),
            "flux_sqv_y": self.zeros(shape=(mi, mj, mk)),
            "flux_sqc_x": self.zeros(shape=(mi, mj, mk)),
            "flux_sqc_y": self.zeros(shape=(mi, mj, mk)),
            "flux_sqr_x": self.zeros(shape=(mi, mj, mk)),
            "flux_sqr_y": self.zeros(shape=(mi, mj, mk)),
        }

        if qv_tnd is not None:
            stencil_args["qv_tnd"] = qv_tnd
        if qc_tnd is not None:
            stencil_args["qc_tnd"] = qc_tnd
        if qr_tnd is not None:
            stencil_args["qr_tnd"] = qr_tnd

        self.stencil_moist(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(mi - 2 * nb, mj - 2 * nb, mk)
        )

        return_list_names = [
            "flux_sqv_x",
            "flux_sqv_y",
            "flux_sqc_x",
            "flux_sqc_y",
            "flux_sqr_x",
            "flux_sqr_y",
        ]
        return_list = tuple(stencil_args[name] for name in return_list_names)

        return return_list

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil_dry")
    def stencil_dry_numpy(
        self,
        s,
        u,
        v,
        su,
        sv,
        flux_s_x,
        flux_s_y,
        flux_su_x,
        flux_su_y,
        flux_sv_x,
        flux_sv_y,
        mtg=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        *,
        dt=0.0,
        dx=0.0,
        dy=0.0,
        origin,
        domain,
        **kwargs
    ):
        ijk = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        ij_ext = tuple(
            slice(o - self.core.extent, o + d + self.core.extent)
            for o, d in zip(origin[:2], domain[:2])
        )
        ijk_x = (ijk[0], ij_ext[1], ijk[2])
        ijk_y = (ij_ext[0], ijk[1], ijk[2])
        ijk_ext = (ij_ext[0], ij_ext[1], ijk[2])

        (
            flux_s_x[ijk_x],
            flux_s_y[ijk_y],
            flux_su_x[ijk_x],
            flux_su_y[ijk_y],
            flux_sv_x[ijk_x],
            flux_sv_y[ijk_y],
        ) = self.core.stencil_subroutine("flux_dry")(
            dt,
            dx,
            dy,
            s[ijk_ext],
            u[ijk_ext],
            v[ijk_ext],
            su[ijk_ext],
            sv[ijk_ext],
            mtg[ijk_ext] if mtg else None,
            s_tnd[ijk_ext] if s_tnd else None,
            su_tnd[ijk_ext] if su_tnd else None,
            sv_tnd[ijk_ext] if sv_tnd else None,
        )

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil_moist")
    def stencil_moist_numpy(
        self,
        s,
        u,
        v,
        sqv,
        sqc,
        sqr,
        flux_sqv_x,
        flux_sqv_y,
        flux_sqc_x,
        flux_sqc_y,
        flux_sqr_x,
        flux_sqr_y,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
        *,
        dt=0.0,
        dx=0.0,
        dy=0.0,
        origin,
        domain,
        **kwargs
    ):
        ijk = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        ij_ext = tuple(
            slice(o - self.core.extent, o + d + self.core.extent)
            for o, d in zip(origin[:2], domain[:2])
        )
        ijk_x = (ijk[0], ij_ext[1], ijk[2])
        ijk_y = (ij_ext[0], ijk[1], ijk[2])
        ijk_ext = (ij_ext[0], ij_ext[1], ijk[2])

        (
            flux_sqv_x[ijk_x],
            flux_sqv_y[ijk_y],
            flux_sqc_x[ijk_x],
            flux_sqc_y[ijk_y],
            flux_sqr_x[ijk_x],
            flux_sqr_y[ijk_y],
        ) = self.core.stencil_subroutine("flux_moist")(
            dt,
            dx,
            dy,
            s[ijk_ext],
            u[ijk_ext],
            v[ijk_ext],
            sqv[ijk_ext],
            sqc[ijk_ext],
            sqr[ijk_ext],
            qv_tnd[ijk_ext] if qv_tnd else None,
            qc_tnd[ijk_ext] if qc_tnd else None,
            qr_tnd[ijk_ext] if qr_tnd else None,
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil_dry")
    def stencil_dry_gt4py(
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        flux_s_x: gtscript.Field["dtype"],
        flux_s_y: gtscript.Field["dtype"],
        flux_su_x: gtscript.Field["dtype"],
        flux_su_y: gtscript.Field["dtype"],
        flux_sv_x: gtscript.Field["dtype"],
        flux_sv_y: gtscript.Field["dtype"],
        mtg: gtscript.Field["dtype"] = None,
        s_tnd: gtscript.Field["dtype"] = None,
        su_tnd: gtscript.Field["dtype"] = None,
        sv_tnd: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0
    ):
        from __externals__ import get_flux_dry

        with computation(PARALLEL), interval(...):
            (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
            ) = get_flux_dry(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u=u,
                v=v,
                mtg=mtg,
                su=su,
                sv=sv,
                s_tnd=s_tnd,
                su_tnd=su_tnd,
                sv_tnd=sv_tnd,
            )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil_moist")
    def stencil_moist_gt4py(
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        flux_sqv_x: gtscript.Field["dtype"],
        flux_sqv_y: gtscript.Field["dtype"],
        flux_sqc_x: gtscript.Field["dtype"],
        flux_sqc_y: gtscript.Field["dtype"],
        flux_sqr_x: gtscript.Field["dtype"],
        flux_sqr_y: gtscript.Field["dtype"],
        qv_tnd: gtscript.Field["dtype"] = None,
        qc_tnd: gtscript.Field["dtype"] = None,
        qr_tnd: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0
    ):
        from __externals__ import get_flux_moist

        with computation(PARALLEL), interval(...):
            (
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ) = get_flux_moist(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u=u,
                v=v,
                sqv=sqv,
                sqc=sqc,
                sqr=sqr,
                qv_tnd=qv_tnd,
                qc_tnd=qc_tnd,
                qr_tnd=qr_tnd,
            )


def get_upwind_fluxes(u, v, phi):
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]

    fx = deepcopy(phi)
    fy = deepcopy(phi)

    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(0, nz):
                fx[i, j, k] = u[i, j, k] * (
                    phi[i - 1, j, k] if u[i, j, k] > 0 else phi[i, j, k]
                )
                fy[i, j, k] = v[i, j, k] * (
                    phi[i, j - 1, k] if v[i, j, k] > 0 else phi[i, j, k]
                )

    return fx, fy


def get_centered_fluxes(u, v, phi):
    fx = deepcopy(phi)
    fy = deepcopy(phi)

    istop = u.shape[0] - 1
    jstop = v.shape[1] - 1

    fx[1:istop, :] = u[1:-1, :] * 0.5 * (phi[: istop - 1, :] + phi[1:istop, :])
    fy[:, 1:jstop] = v[:, 1:-1] * 0.5 * (phi[:, : jstop - 1] + phi[:, 1:jstop])

    return fx, fy


def get_third_order_upwind_fluxes(u, v, phi):
    f4x = deepcopy(phi)
    f4y = deepcopy(phi)

    istop = u.shape[0] - 2
    jstop = v.shape[1] - 2

    f4x[2:istop, :] = (
        u[2:-2, :]
        / 12.0
        * (
            7.0 * (phi[2:istop, :] + phi[1 : istop - 1, :])
            - (phi[3 : istop + 1, :] + phi[: istop - 2, :])
        )
    )
    f4y[:, 2:jstop] = (
        v[:, 2:-2]
        / 12.0
        * (
            7.0 * (phi[:, 2:jstop] + phi[:, 1 : jstop - 1])
            - (phi[:, 3 : jstop + 1] + phi[:, : jstop - 2])
        )
    )

    fx = deepcopy(phi)
    fy = deepcopy(phi)

    fx[2:istop, :] = f4x[2:istop, :] - np.abs(u[2:-2, :]) / 12.0 * (
        3.0 * (phi[2:istop, :] - phi[1 : istop - 1, :])
        - (phi[3 : istop + 1, :] - phi[: istop - 2, :])
    )
    fy[:, 2:jstop] = f4y[:, 2:jstop] - np.abs(v[:, 2:-2]) / 12.0 * (
        3.0 * (phi[:, 2:jstop] - phi[:, 1 : jstop - 1])
        - (phi[:, 3 : jstop + 1] - phi[:, : jstop - 2])
    )

    return fx, fy


def get_fifth_order_upwind_fluxes(u, v, phi):
    f6x = deepcopy(phi)
    f6y = deepcopy(phi)

    istop = u.shape[0] - 3
    jstop = v.shape[1] - 3

    f6x[3:istop, :] = (
        u[3:-3, :]
        / 60.0
        * (
            37.0 * (phi[3:istop, :] + phi[2 : istop - 1, :])
            - 8.0 * (phi[4 : istop + 1, :] + phi[1 : istop - 2, :])
            + (phi[5 : istop + 2, :] + phi[: istop - 3, :])
        )
    )
    f6y[:, 3:jstop] = (
        v[:, 3:-3]
        / 60.0
        * (
            37.0 * (phi[:, 3:jstop] + phi[:, 2 : jstop - 1])
            - 8.0 * (phi[:, 4 : jstop + 1] + phi[:, 1 : jstop - 2])
            + (phi[:, 5 : jstop + 2] + phi[:, : jstop - 3])
        )
    )

    fx = deepcopy(phi)
    fy = deepcopy(phi)

    fx[3:istop, :] = f6x[3:istop, :] - np.abs(u[3:-3, :]) / 60.0 * (
        10.0 * (phi[3:istop, :] - phi[2 : istop - 1, :])
        - 5.0 * (phi[4 : istop + 1, :] - phi[1 : istop - 2, :])
        + (phi[5 : istop + 2, :] - phi[: istop - 3, :])
    )
    fy[:, 3:jstop] = f6y[:, 3:jstop] - np.abs(v[:, 3:-3]) / 60.0 * (
        10.0 * (phi[:, 3:jstop] - phi[:, 2 : jstop - 1])
        - 5.0 * (phi[:, 4 : jstop + 1] - phi[:, 1 : jstop - 2])
        + (phi[:, 5 : jstop + 2] - phi[:, : jstop - 3])
    )

    return fx, fy


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
    cls,
    flux_scheme,
    domain,
    field,
    dt,
    backend,
    backend_options,
    storage_options,
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    nb = domain.horizontal_boundary.nb
    flux_type = flux_properties[flux_scheme]["type"]
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    ws = WrappingStencil(
        cls, flux_scheme, nb, backend, backend_options, storage_options
    )

    s = ws.zeros(shape=(nx + 1, ny + 1, nz))
    s[...] = field[: nx + 1, : ny + 1, :nz]
    u = ws.zeros(shape=(nx + 1, ny + 1, nz))
    u[...] = field[1 : nx + 2, : ny + 1, :nz]
    v = ws.zeros(shape=(nx + 1, ny + 1, nz))
    v[...] = field[: nx + 1, 1 : ny + 2, :nz]
    su = ws.zeros(shape=(nx + 1, ny + 1, nz))
    su[...] = field[1 : nx + 2, : ny + 1, :nz]
    sv = ws.zeros(shape=(nx + 1, ny + 1, nz))
    sv[...] = field[1 : nx + 2, 1 : ny + 2, :nz]
    sqv = ws.zeros(shape=(nx + 1, ny + 1, nz))
    sqv[...] = field[: nx + 1, : ny + 1, 1 : nz + 1]
    sqc = ws.zeros(shape=(nx + 1, ny + 1, nz))
    sqc[...] = field[1 : nx + 2, : ny + 1, 1 : nz + 1]
    sqr = ws.zeros(shape=(nx + 1, ny + 1, nz))
    sqr[...] = field[1 : nx + 2, 1 : ny + 2, 1 : nz + 1]

    #
    # dry
    #
    fsx, fsy, fsux, fsuy, fsvx, fsvy = ws.call_dry(dt, dx, dy, s, u, v, su, sv)

    flux_s_x, flux_s_y = get_fluxes(u, v, s)
    x = slice(nb, grid.nx + 1 - nb)
    y = slice(nb, grid.ny + 1 - nb)
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
    fsqvx, fsqvy, fsqcx, fsqcy, fsqrx, fsqry = ws.call_moist(
        dt, dx, dy, s, u, v, sqv, sqc, sqr
    )

    flux_sqv_x, flux_sqv_y = get_fluxes(u, v, sqv)
    compare_arrays(fsqvx[x, y], flux_sqv_x[x, y])
    compare_arrays(fsqvy[x, y], flux_sqv_y[x, y])

    flux_sqc_x, flux_sqc_y = get_fluxes(u, v, sqc)
    compare_arrays(fsqcx[x, y], flux_sqc_x[x, y])
    compare_arrays(fsqcy[x, y], flux_sqc_y[x, y])

    flux_sqr_x, flux_sqr_y = get_fluxes(u, v, sqr)
    compare_arrays(fsqrx[x, y], flux_sqr_x[x, y])
    compare_arrays(fsqry[x, y], flux_sqr_y[x, y])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicHorizontalFlux, "upwind", domain, field, dt, backend, bo, so
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_centered(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicHorizontalFlux,
        "centered",
        domain,
        field,
        dt,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_third_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicHorizontalFlux,
        "third_order_upwind",
        domain,
        field,
        dt,
        backend,
        bo,
        so,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_fifth_order_upwind(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 2, ny + 2, nz + 1),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicHorizontalFlux,
        "fifth_order_upwind",
        domain,
        field,
        dt,
        backend,
        bo,
        so,
    )


if __name__ == "__main__":
    pytest.main([__file__])
