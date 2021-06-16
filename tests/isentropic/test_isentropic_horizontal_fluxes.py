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

from tasmania.python.framework.generic_functions import to_numpy
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

from tests import conf
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


def test_registry():
    registry = IsentropicHorizontalFlux.registry[
        "tasmania.python.isentropic.dynamics.horizontal_fluxes."
        "IsentropicHorizontalFlux"
    ]

    assert "upwind" in registry
    assert registry["upwind"] == Upwind
    assert "centered" in registry
    assert registry["centered"] == Centered
    assert "third_order_upwind" in registry
    assert registry["third_order_upwind"] == ThirdOrderUpwind
    assert "fifth_order_upwind" in registry
    assert registry["fifth_order_upwind"] == FifthOrderUpwind


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

        bo = deepcopy(self.backend_options)
        bo.externals = {
            name: self.core.compile_subroutine(name)
            for name in self.core.external_names
        }
        get_flux_dry = self.core.compile_subroutine(
            "flux_dry", backend_options=bo
        )
        get_flux_moist = self.core.compile_subroutine(
            "flux_moist", backend_options=bo
        )
        self.backend_options.externals = bo.externals.copy()
        self.backend_options.externals["extent"] = self.core.extent
        self.backend_options.externals.update(
            {
                "get_flux_dry": get_flux_dry,
                "get_flux_moist": get_flux_moist,
            }
        )
        self.stencil_dry = self.compile_stencil("stencil_dry")
        self.stencil_moist = self.compile_stencil("stencil_moist")

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

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="stencil_dry"
    )
    def stencil_dry_numpy(
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
        domain
    ):
        ijk = (
            slice(origin[0], origin[0] + domain[0]),
            slice(origin[1], origin[1] + domain[1]),
            slice(origin[2], origin[2] + domain[2]),
        )
        ij_ext = (
            slice(origin[0] - extent, origin[0] + domain[0] + extent),
            slice(origin[1] - extent, origin[1] + domain[1] + extent),
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
        ) = get_flux_dry(
            dt,
            dx,
            dy,
            s[ijk_ext],
            u[ijk_ext],
            v[ijk_ext],
            su[ijk_ext],
            sv[ijk_ext],
            mtg[ijk_ext] if mtg is not None else None,
            s_tnd[ijk_ext] if s_tnd is not None else None,
            su_tnd[ijk_ext] if su_tnd is not None else None,
            sv_tnd[ijk_ext] if sv_tnd is not None else None,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="stencil_moist"
    )
    def stencil_moist_numpy(
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
        domain
    ):
        ijk = (
            slice(origin[0], origin[0] + domain[0]),
            slice(origin[1], origin[1] + domain[1]),
            slice(origin[2], origin[2] + domain[2]),
        )
        ij_ext = (
            slice(origin[0] - extent, origin[0] + domain[0] + extent),
            slice(origin[1] - extent, origin[1] + domain[1] + extent),
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
        ) = get_flux_moist(
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

    fx = np.zeros_like(phi)
    fy = np.zeros_like(phi)

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
    fx = np.zeros_like(phi)
    fy = np.zeros_like(phi)

    istop = u.shape[0] - 1
    jstop = v.shape[1] - 1

    fx[1:istop, :] = u[1:-1, :] * 0.5 * (phi[: istop - 1, :] + phi[1:istop, :])
    fy[:, 1:jstop] = v[:, 1:-1] * 0.5 * (phi[:, : jstop - 1] + phi[:, 1:jstop])

    return fx, fy


def get_third_order_upwind_fluxes(u, v, phi):
    f4x = np.zeros_like(phi)
    f4y = np.zeros_like(phi)

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

    fx = np.zeros_like(phi)
    fy = np.zeros_like(phi)

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
    f6x = np.zeros_like(phi)
    f6y = np.zeros_like(phi)

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

    fx = np.zeros_like(phi)
    fy = np.zeros_like(phi)

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

    u_np, v_np, s_np = to_numpy(u), to_numpy(v), to_numpy(s)
    flux_s_x, flux_s_y = get_fluxes(u_np, v_np, s_np)
    x = slice(nb, grid.nx + 1 - nb)
    y = slice(nb, grid.ny + 1 - nb)
    z = slice(0, grid.nz)
    compare_arrays(fsx, flux_s_x, slice=(x, y, z))
    compare_arrays(fsy, flux_s_y, slice=(x, y, z))

    su_np = to_numpy(su)
    flux_su_x, flux_su_y = get_fluxes(u_np, v_np, su_np)
    compare_arrays(fsux, flux_su_x, slice=(x, y, z))
    compare_arrays(fsuy, flux_su_y, slice=(x, y, z))

    sv_np = to_numpy(sv)
    flux_sv_x, flux_sv_y = get_fluxes(u_np, v_np, sv_np)
    compare_arrays(fsvx, flux_sv_x, slice=(x, y, z))
    compare_arrays(fsvy, flux_sv_y, slice=(x, y, z))

    #
    # moist
    #
    fsqvx, fsqvy, fsqcx, fsqcy, fsqrx, fsqry = ws.call_moist(
        dt, dx, dy, s, u, v, sqv, sqc, sqr
    )

    sqv_np = to_numpy(sqv)
    flux_sqv_x, flux_sqv_y = get_fluxes(u_np, v_np, sqv_np)
    compare_arrays(fsqvx, flux_sqv_x, slice=(x, y, z))
    compare_arrays(fsqvy, flux_sqv_y, slice=(x, y, z))

    sqc_np = to_numpy(sqc)
    flux_sqc_x, flux_sqc_y = get_fluxes(u_np, v_np, sqc_np)
    compare_arrays(fsqcx, flux_sqc_x, slice=(x, y, z))
    compare_arrays(fsqcy, flux_sqc_y, slice=(x, y, z))

    sqr_np = to_numpy(sqr)
    flux_sqr_x, flux_sqr_y = get_fluxes(u_np, v_np, sqr_np)
    compare_arrays(fsqrx, flux_sqr_x, slice=(x, y, z))
    compare_arrays(fsqry, flux_sqr_y, slice=(x, y, z))


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("flux_scheme", flux_properties.keys())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, flux_scheme, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="field",
    )

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    validation(
        IsentropicHorizontalFlux,
        flux_scheme,
        domain,
        field,
        dt,
        backend,
        bo,
        so,
    )


if __name__ == "__main__":
    # pytest.main([__file__])
    test("centered", "numba:cpu:numpy", float)
