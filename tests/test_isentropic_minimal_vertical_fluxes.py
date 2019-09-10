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
    assume,
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
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.isentropic.dynamics.implementations.minimal_vertical_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import compare_arrays, st_domain, st_floats, st_one_of
except (ModuleNotFoundError, ImportError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import compare_arrays, st_domain, st_floats, st_one_of


class WrappingStencil:
    def __init__(self, core, backend, dtype, halo, module, rebuild):
        self.core = core
        self.backend = backend
        self.dtype = dtype
        self.halo = halo
        self.module = module
        self.rebuild = rebuild

    def __call__(self, dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        mi, mj, mk = s.shape
        descriptor = get_storage_descriptor((mi, mj, mk), self.dtype, halo=self.halo)

        stencil_args = {
            "w": gt.storage.from_array(w, descriptor, backend=self.backend),
            "s": gt.storage.from_array(s, descriptor, backend=self.backend),
            "su": gt.storage.from_array(su, descriptor, backend=self.backend),
            "sv": gt.storage.from_array(sv, descriptor, backend=self.backend),
            "flux_s": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_su": gt.storage.zeros(descriptor, backend=self.backend),
            "flux_sv": gt.storage.zeros(descriptor, backend=self.backend),
        }
        moist = sqv is not None
        if moist:
            stencil_args.update(
                {
                    "sqv": gt.storage.from_array(sqv, descriptor, backend=self.backend),
                    "sqc": gt.storage.from_array(sqc, descriptor, backend=self.backend),
                    "sqr": gt.storage.from_array(sqr, descriptor, backend=self.backend),
                    "flux_sqv": gt.storage.zeros(descriptor, backend=self.backend),
                    "flux_sqc": gt.storage.zeros(descriptor, backend=self.backend),
                    "flux_sqr": gt.storage.zeros(descriptor, backend=self.backend),
                }
            )

        externals = self.core.externals.copy()
        externals["core"] = self.core.__call__
        externals["moist"] = moist

        decorator = gt.stencil(
            self.backend,
            externals=externals,
            rebuild=self.rebuild,
            min_signature=True,
            module=self.module,
        )
        stencil = decorator(self.stencil_defs)

        nb = self.core.extent
        stencil(
            **stencil_args,
            dt=dt,
            dz=dz,
            origin={"_all_": (0, 0, nb)},
            domain=(mi, mj, mk - 2 * nb)
        )

        names = ["flux_s", "flux_su", "flux_sv"]
        if moist:
            names += ["flux_sqv", "flux_sqc", "flux_sqr"]
        return_list = tuple(stencil_args[name].data for name in names)

        return return_list

    @staticmethod
    def stencil_defs(
        w: gt.storage.f64_sd,
        s: gt.storage.f64_sd,
        su: gt.storage.f64_sd,
        sv: gt.storage.f64_sd,
        sqv: gt.storage.f64_sd,
        sqc: gt.storage.f64_sd,
        sqr: gt.storage.f64_sd,
        flux_s: gt.storage.f64_sd,
        flux_su: gt.storage.f64_sd,
        flux_sv: gt.storage.f64_sd,
        flux_sqv: gt.storage.f64_sd,
        flux_sqc: gt.storage.f64_sd,
        flux_sqr: gt.storage.f64_sd,
        *,
        dt: float,
        dz: float
    ):
        if not moist:
            flux_s, flux_su, flux_sv = core(dt=dt, dz=dz, w=w, s=s, su=su, sv=sv)
        else:
            flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = core(
                dt=dt, dz=dz, w=w, s=s, su=su, sv=sv, sqv=sqv, sqc=sqc, sqr=sqr
            )


def get_upwind_flux(w, phi):
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]

    f = np.zeros_like(phi, dtype=phi.dtype)

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                f[i, j, k] = w[i, j, k] * (
                    phi[i, j, k] if w[i, j, k] > 0 else phi[i, j, k - 1]
                )

    return f


def get_centered_flux(w, phi):
    f = np.zeros_like(phi, dtype=phi.dtype)

    kstop = w.shape[2] - 1

    f[:, :, 1:kstop] = w[:, :, 1:-1] * 0.5 * (phi[:, :, : kstop - 1] + phi[:, :, 1:kstop])

    return f


def get_third_order_upwind_flux(w, phi):
    f4 = np.zeros_like(phi, dtype=phi.dtype)

    kstop = w.shape[2] - 2

    f4[:, :, 2:kstop] = (
        w[:, :, 2:-2]
        / 12.0
        * (
            7.0 * (phi[:, :, 1 : kstop - 1] + phi[:, :, 2:kstop])
            - (phi[:, :, : kstop - 2] + phi[:, :, 3 : kstop + 1])
        )
    )

    f = np.zeros_like(phi, dtype=phi.dtype)

    f[:, :, 2:kstop] = f4[:, :, 2:kstop] - np.abs(w[:, :, 2:-2]) / 12.0 * (
        3.0 * (phi[:, :, 1 : kstop - 1] - phi[:, :, 2:kstop])
        - (phi[:, :, : kstop - 2] - phi[:, :, 3 : kstop + 1])
    )

    return f


def get_fifth_order_upwind_flux(w, phi):
    f6 = np.zeros_like(phi, dtype=phi.dtype)

    kstop = w.shape[2] - 3

    f6[:, :, 3:kstop] = (
        w[:, :, 3:-3]
        / 60.0
        * (
            37.0 * (phi[:, :, 2 : kstop - 1] + phi[:, :, 3:kstop])
            - 8.0 * (phi[:, :, 1 : kstop - 2] + phi[:, :, 4 : kstop + 1])
            + (phi[:, :, : kstop - 3] + phi[:, :, 5 : kstop + 2])
        )
    )

    f = np.zeros_like(phi, dtype=phi.dtype)

    f[:, :, 3:kstop] = f6[:, :, 3:kstop] - np.abs(w[:, :, 3:-3]) / 60.0 * (
        10.0 * (phi[:, :, 2 : kstop - 1] - phi[:, :, 3:kstop])
        - 5.0 * (phi[:, :, 1 : kstop - 2] - phi[:, :, 4 : kstop + 1])
        + (phi[:, :, : kstop - 3] - phi[:, :, 5 : kstop + 2])
    )

    return f


flux_properties = {
    "upwind": {"type": Upwind, "get_fluxes": get_upwind_flux},
    "centered": {"type": Centered, "get_fluxes": get_centered_flux},
    "third_order_upwind": {
        "type": ThirdOrderUpwind,
        "get_fluxes": get_third_order_upwind_flux,
    },
    "fifth_order_upwind": {
        "type": FifthOrderUpwind,
        "get_fluxes": get_fifth_order_upwind_flux,
    },
}


def validation_dry(
    flux_scheme, domain, field, timestep, backend, dtype, halo, module, rebuild
):
    grid = domain.numerical_grid
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    w = field[1:, 1:, :-1]
    s = field[:-1, :-1, :-1]
    su = field[1:, :-1, :-1]
    sv = field[:-1, :-1, :-1]

    core = IsentropicMinimalVerticalFlux.factory(flux_scheme)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, backend, dtype, halo, module, rebuild)

    z = slice(nb, grid.nz - nb + 1)
    dz = grid.dz.to_units("K").values.item()

    fs, fsu, fsv = ws(timestep, dz, w, s, su, sv)

    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, z], flux_s[:, :, z])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, z], flux_su[:, :, z])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, z], flux_sv[:, :, z])


def validation_moist(
    flux_scheme, domain, field, timestep, backend, dtype, halo, module, rebuild
):
    grid = domain.numerical_grid
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    w = field[1:, 1:, :-1]
    s = field[:-1, :-1, :-1]
    su = field[1:, :-1, :-1]
    sv = field[:-1, :-1, :-1]
    sqv = field[:-1, :-1, 1:]
    sqc = field[1:, :-1, 1:]
    sqr = field[:-1, :-1, 1:]

    core = IsentropicMinimalVerticalFlux.factory(flux_scheme)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, backend, dtype, halo, module, rebuild)

    z = slice(nb, grid.nz - nb + 1)
    dz = grid.dz.to_units("K").values.item()

    fs, fsu, fsv, fsqv, fsqc, fsqr = ws(
        timestep, dz, w, s, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
    )

    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, z], flux_s[:, :, z])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, z], flux_su[:, :, z])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, z], flux_sv[:, :, z])

    flux_sqv = get_fluxes(w, sqv)
    compare_arrays(fsqv[:, :, z], flux_sqv[:, :, z])

    flux_sqc = get_fluxes(w, sqc)
    compare_arrays(fsqc[:, :, z], flux_sqc[:, :, z])

    flux_sqr = get_fluxes(w, sqr)
    compare_arrays(fsqr[:, :, z], flux_sqr[:, :, z])


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
    domain = data.draw(st_domain(zaxis_length=(3, 40)), label="domain")
    grid = domain.physical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 1, grid.ny + 1, grid.nz + 2),
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
    validation_dry(
        "upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="upwind_dry",
        rebuild=False,
    )
    validation_moist(
        "upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="upwind_moist",
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
def test_centered(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 40)), label="domain")
    grid = domain.physical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 1, grid.ny + 1, grid.nz + 2),
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
    validation_dry(
        "centered",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="centered_dry",
        rebuild=False,
    )
    validation_moist(
        "centered",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="centered_moist",
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
def test_third_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(5, 40)), label="domain")
    grid = domain.physical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 1, grid.ny + 1, grid.nz + 2),
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
    validation_dry(
        "third_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="third_order_upwind_dry",
        rebuild=False,
    )
    validation_moist(
        "third_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="third_order_upwind_moist",
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
    domain = data.draw(st_domain(zaxis_length=(7, 40)), label="domain")
    grid = domain.physical_grid
    dtype = grid.x.dtype
    field = data.draw(
        st_arrays(
            dtype,
            (grid.nx + 1, grid.ny + 1, grid.nz + 2),
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
    validation_dry(
        "fifth_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="fifth_order_upwind_dry",
        rebuild=False,
    )
    validation_moist(
        "fifth_order_upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        halo,
        module="fifth_order_upwind_moist",
        rebuild=False,
    )


if __name__ == "__main__":
    pytest.main([__file__])
