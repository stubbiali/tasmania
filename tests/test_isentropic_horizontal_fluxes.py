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
from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.implementations.horizontal_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import st_domain, st_floats, st_one_of, compare_arrays
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import st_domain, st_floats, st_one_of, compare_arrays


class WrappingStencil:
    def __init__(self, core, nb, backend, dtype, halo, rebuild):
        self.core = core
        self.nb = nb
        self.backend = backend
        self.dtype = dtype
        self.halo = halo
        self.rebuild = rebuild

    def __call__(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
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
        descriptor = get_storage_descriptor((mi, mj, mk), self.dtype, halo=self.halo)

        stencil_args = {
            "s": gt.storage.from_array(s, descriptor, backend=self.backend),
            "u": gt.storage.from_array(u, descriptor, backend=self.backend),
            "v": gt.storage.from_array(v, descriptor, backend=self.backend),
            "mtg": gt.storage.from_array(mtg, descriptor, backend=self.backend),
            "su": gt.storage.from_array(su, descriptor, backend=self.backend),
            "sv": gt.storage.from_array(sv, descriptor, backend=self.backend),
            "flux_s_x": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_s_y": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_su_x": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_su_y": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_sv_x": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_sv_y": gt.storage.zeros(descriptor, backend=self.backend),
        }
        moist = sqv is not None
        if moist:
            stencil_args["sqv"] = gt.storage.from_array(
                sqv, descriptor, backend=self.backend
            )
            stencil_args["flux_sqv_x"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
            stencil_args["flux_sqv_y"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
            stencil_args["sqc"] = gt.storage.from_array(
                sqc, descriptor, backend=self.backend
            )
            stencil_args["flux_sqc_x"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
            stencil_args["flux_sqc_y"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
            stencil_args["sqr"] = gt.storage.from_array(
                sqr, descriptor, backend=self.backend
            )
            stencil_args["flux_sqr_x"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
            stencil_args["flux_sqr_y"] = gt.storage.zeros(
                descriptor, backend=self.backend
            )
        s_tnd_on = s_tnd is not None
        stencil_args["s_tnd"] = (
            stencil_args["s"]
            if not s_tnd_on
            else gt.storage.from_array(s_tnd, descriptor, backend=self.backend)
        )
        su_tnd_on = su_tnd is not None
        stencil_args["su_tnd"] = (
            stencil_args["su"]
            if not su_tnd_on
            else gt.storage.from_array(su_tnd, descriptor, backend=self.backend)
        )
        sv_tnd_on = sv_tnd is not None
        stencil_args["sv_tnd"] = (
            stencil_args["sv"]
            if not sv_tnd_on
            else gt.storage.from_array(sv_tnd, descriptor, backend=self.backend)
        )
        qv_tnd_on = qv_tnd is not None
        if moist:
            stencil_args["qv_tnd"] = (
                stencil_args["sqv"]
                if not qv_tnd_on
                else gt.storage.from_array(qv_tnd, descriptor, backend=self.backend)
            )
        qc_tnd_on = qc_tnd is not None
        if moist:
            stencil_args["qc_tnd"] = (
                stencil_args["sqc"]
                if not qc_tnd_on
                else gt.storage.from_array(qc_tnd, descriptor, backend=self.backend)
            )
        qr_tnd_on = qr_tnd is not None
        if moist:
            stencil_args["qr_tnd"] = (
                stencil_args["sqr"]
                if not qr_tnd_on
                else gt.storage.from_array(qr_tnd, descriptor, backend=self.backend)
            )

        externals = self.core.externals.copy()
        externals.update(
            {
                "core": self.core.__call__,
                "moist": moist,
                "s_tnd_on": s_tnd_on,
                "su_tnd_on": su_tnd_on,
                "sv_tnd_on": sv_tnd_on,
                "qv_tnd_on": qv_tnd_on,
                "qc_tnd_on": qc_tnd_on,
                "qr_tnd_on": qr_tnd_on,
            }
        )

        decorator = gt.stencil(
            self.backend, externals=externals, rebuild=self.rebuild, min_signature=True
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
        return_list = tuple(stencil_args[name].data for name in return_list_names)

        return return_list

    @staticmethod
    def stencil_defs(
        s: gt.storage.f64_sd,
        u: gt.storage.f64_sd,
        v: gt.storage.f64_sd,
        mtg: gt.storage.f64_sd,
        su: gt.storage.f64_sd,
        sv: gt.storage.f64_sd,
        sqv: gt.storage.f64_sd,
        sqc: gt.storage.f64_sd,
        sqr: gt.storage.f64_sd,
        s_tnd: gt.storage.f64_sd,
        su_tnd: gt.storage.f64_sd,
        sv_tnd: gt.storage.f64_sd,
        qv_tnd: gt.storage.f64_sd,
        qc_tnd: gt.storage.f64_sd,
        qr_tnd: gt.storage.f64_sd,
        flux_s_x: gt.storage.f64_sd,
        flux_s_y: gt.storage.f64_sd,
        flux_su_x: gt.storage.f64_sd,
        flux_su_y: gt.storage.f64_sd,
        flux_sv_x: gt.storage.f64_sd,
        flux_sv_y: gt.storage.f64_sd,
        flux_sqv_x: gt.storage.f64_sd,
        flux_sqv_y: gt.storage.f64_sd,
        flux_sqc_x: gt.storage.f64_sd,
        flux_sqc_y: gt.storage.f64_sd,
        flux_sqr_x: gt.storage.f64_sd,
        flux_sqr_y: gt.storage.f64_sd,
        *,
        dt: float,
        dx: float,
        dy: float
    ):
        if not moist:
            flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = core(
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
        else:
            flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = core(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u=u,
                v=v,
                mtg=mtg,
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


def get_upwind_fluxes(u, v, phi):
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]

    fx = np.zeros_like(phi, dtype=phi.dtype)
    fy = np.zeros_like(phi, dtype=phi.dtype)

    for i in range(0, nx - 1):
        for j in range(0, ny - 1):
            for k in range(0, nz):
                fx[i, j, k] = u[i + 1, j, k] * (
                    phi[i, j, k] if u[i + 1, j, k] > 0 else phi[i + 1, j, k]
                )
                fy[i, j, k] = v[i, j + 1, k] * (
                    phi[i, j, k] if v[i, j + 1, k] > 0 else phi[i, j + 1, k]
                )

    return fx, fy


def get_centered_fluxes(u, v, phi):
    fx = np.zeros_like(phi, dtype=phi.dtype)
    fy = np.zeros_like(phi, dtype=phi.dtype)

    istop = u.shape[0] - 2
    jstop = v.shape[1] - 2

    fx[:istop, :] = u[1:-1, :] * 0.5 * (phi[:istop, :] + phi[1 : 1 + istop, :])
    fy[:, :jstop] = v[:, 1:-1] * 0.5 * (phi[:, :jstop] + phi[:, 1 : jstop + 1])

    return fx, fy


def get_maccormack_fluxes():
    ### TODO ###
    pass


def get_third_order_upwind_fluxes(u, v, phi):
    f4x = np.zeros_like(phi, dtype=phi.dtype)
    f4y = np.zeros_like(phi, dtype=phi.dtype)

    istop = u.shape[0] - 3
    jstop = v.shape[1] - 3

    f4x[1:istop, :] = (
        u[2:-2, :]
        / 12.0
        * (
            7.0 * (phi[2 : istop + 1, :] + phi[1:istop, :])
            - (phi[3 : istop + 2, :] + phi[: istop - 1, :])
        )
    )
    f4y[:, 1:jstop] = (
        v[:, 2:-2]
        / 12.0
        * (
            7.0 * (phi[:, 2 : jstop + 1] + phi[:, 1:jstop])
            - (phi[:, 3 : jstop + 1] + phi[:, : jstop - 1])
        )
    )

    fx = np.zeros_like(phi, dtype=phi.dtype)
    fy = np.zeros_like(phi, dtype=phi.dtype)

    fx[1:istop, :] = f4x[1:istop, :] - np.abs(u[2:-2, :]) / 12.0 * (
        3.0 * (phi[2 : istop + 1, :] - phi[1:istop, :])
        - (phi[3 : istop + 2, :] - phi[: istop - 1, :])
    )
    fy[:, 1:jstop] = f4y[:, 1:jstop] - np.abs(v[:, 2:-2]) / 12.0 * (
        3.0 * (phi[:, 2 : jstop + 1] - phi[:, 1:jstop])
        - (phi[:, 3 : jstop + 2] - phi[:, : jstop - 1])
    )

    return fx, fy


def get_fifth_order_upwind_fluxes(u, v, phi):
    f6x = np.zeros_like(phi, dtype=phi.dtype)
    f6y = np.zeros_like(phi, dtype=phi.dtype)

    istop = u.shape[0] - 4
    jstop = v.shape[1] - 4

    f6x[2:istop, :] = (
        u[3:-3, :]
        / 60.0
        * (
            37.0 * (phi[3 : istop + 1, :] + phi[2:istop, :])
            - 8.0 * (phi[4 : istop + 2, :] + phi[1 : istop - 1, :])
            + (phi[5 : istop + 3, :] + phi[: istop - 2, :])
        )
    )
    f6y[:, 2:jstop] = (
        v[:, 3:-3]
        / 60.0
        * (
            37.0 * (phi[:, 3 : jstop + 1] + phi[:, 2:jstop])
            - 8.0 * (phi[:, 4 : jstop + 2] + phi[:, 1 : jstop - 1])
            + (phi[:, 5 : jstop + 3] + phi[:, : jstop - 2])
        )
    )

    fx = np.zeros_like(phi, dtype=phi.dtype)
    fy = np.zeros_like(phi, dtype=phi.dtype)

    fx[2:istop, :] = f6x[2:istop, :] - np.abs(u[3:-3, :]) / 60.0 * (
        10.0 * (phi[3 : istop + 1, :] - phi[2:istop, :])
        - 5.0 * (phi[4 : istop + 2, :] - phi[1 : istop - 1, :])
        + (phi[5 : istop + 3, :] - phi[: istop - 2, :])
    )
    fy[:, 2:jstop] = f6y[:, 2:jstop] - np.abs(v[:, 3:-3]) / 60.0 * (
        10.0 * (phi[:, 3 : jstop + 1] - phi[:, 2:jstop])
        - 5.0 * (phi[:, 4 : jstop + 2] - phi[:, 1 : jstop - 1])
        + (phi[:, 5 : jstop + 3] - phi[:, : jstop - 2])
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


def validation(flux_scheme, domain, field, timestep, backend, dtype, halo, rebuild):
    grid = domain.numerical_grid
    nb = domain.horizontal_boundary.nb
    flux_type = flux_properties[flux_scheme]["type"]
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    s = field[:-1, :-1, :-1]
    u = field[:-1, :-1, 1:]
    v = field[:-1, 1:, :-1]
    mtg = field[:-1, 1:, :-1]
    su = field[1:, :-1, :-1]
    sv = field[:-1, :-1, :-1]
    sqv = field[:-1, :-1, 1:]
    sqc = field[1:, :-1, 1:]
    sqr = field[:-1, :-1, 1:]

    core = IsentropicHorizontalFlux.factory(flux_scheme)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, nb, backend, dtype, halo, rebuild=rebuild)

    #
    # dry
    #
    fsx, fsy, fsux, fsuy, fsvx, fsvy = ws(timestep, dx, dy, s, u, v, mtg, su, sv)

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
        timestep, dx, dy, s, u, v, mtg, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
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
def _test_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(),
            fill=hyp_st.nothing(),
        )
    )
    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    validation("upwind", domain, field, timestep, backend, dtype, halo, rebuild=True)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_centered(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(),
            fill=hyp_st.nothing(),
        )
    )
    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    validation("centered", domain, field, timestep, backend, dtype, halo, rebuild=True)


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
def _test_third_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(),
            fill=hyp_st.nothing(),
        )
    )
    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

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
        halo,
        rebuild=True,
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
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 2, grid.ny + 2, grid.nz + 1),
            elements=st_floats(),
            fill=hyp_st.nothing(),
        )
    )
    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

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
        halo,
        rebuild=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
