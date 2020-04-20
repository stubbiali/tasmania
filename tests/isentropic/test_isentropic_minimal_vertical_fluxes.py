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

from gt4py import gtscript, storage as gt_storage
from gt4py.gtscript import PARALLEL, __INLINED, computation, interval

from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.isentropic.dynamics.implementations.minimal_vertical_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import zeros

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import compare_arrays


class WrappingStencil:
    def __init__(self, core, backend, dtype, default_origin, rebuild):
        self.core = core
        self.backend = backend
        self.dtype = dtype
        self.default_origin = default_origin
        self.rebuild = rebuild

    def __call__(self, dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        mi, mj, mk = s.shape
        storage_shape = (mi, mj, mk)

        stencil_args = {
            "w": w,
            "s": s,
            "su": su,
            "sv": sv,
            "flux_s": zeros(
                storage_shape,
                True,
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_su": zeros(
                storage_shape,
                True,
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_sv": zeros(
                storage_shape,
                True,
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
        }
        moist = sqv is not None
        if moist:
            stencil_args.update(
                {
                    "sqv": sqv,
                    "sqc": sqc,
                    "sqr": sqr,
                    "flux_sqv": zeros(
                        storage_shape,
                        True,
                        backend=self.backend,
                        dtype=s.dtype,
                        default_origin=self.default_origin,
                    ),
                    "flux_sqc": zeros(
                        storage_shape,
                        True,
                        backend=self.backend,
                        dtype=s.dtype,
                        default_origin=self.default_origin,
                    ),
                    "flux_sqr": zeros(
                        storage_shape,
                        True,
                        backend=self.backend,
                        dtype=s.dtype,
                        default_origin=self.default_origin,
                    ),
                }
            )

        externals = {
            "core": self.core.call_gt,
            "extent": self.core.extent,
            "moist": moist,
        }

        decorator = gtscript.stencil(
            self.backend,
            dtypes={"dtype": self.dtype},
            externals=externals,
            rebuild=self.rebuild,
        )
        stencil = decorator(self.stencil_defs)

        stencil(
            **stencil_args,
            dt=dt,
            dz=dz,
            origin={"_all_": (0, 0, 0)},
            domain=(mi, mj, mk)
        )

        names = ["flux_s", "flux_su", "flux_sv"]
        if moist:
            names += ["flux_sqv", "flux_sqc", "flux_sqr"]
        return_list = tuple(stencil_args[name] for name in names)

        return return_list

    @staticmethod
    def stencil_defs(
        w: gtscript.Field["dtype"],
        s: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        flux_s: gtscript.Field["dtype"],
        flux_su: gtscript.Field["dtype"],
        flux_sv: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"] = None,
        sqc: gtscript.Field["dtype"] = None,
        sqr: gtscript.Field["dtype"] = None,
        flux_sqv: gtscript.Field["dtype"] = None,
        flux_sqc: gtscript.Field["dtype"] = None,
        flux_sqr: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dz: float = 0.0
    ):
        from __externals__ import core, extent, moist

        with computation(PARALLEL), interval(0, extent):
            flux_s = 0.0
            flux_su = 0.0
            flux_sv = 0.0
            if __INLINED(moist):
                flux_sqv = 0.0
                flux_sqc = 0.0
                flux_sqr = 0.0

        with computation(PARALLEL), interval(extent, -extent):
            if __INLINED(not moist):
                flux_s, flux_su, flux_sv = core(dt=dt, dz=dz, w=w, s=s, su=su, sv=sv)
            else:
                flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = core(
                    dt=dt, dz=dz, w=w, s=s, su=su, sv=sv, sqv=sqv, sqc=sqc, sqr=sqr
                )

        with computation(PARALLEL), interval(-extent, None):
            flux_s = 0.0
            flux_su = 0.0
            flux_sv = 0.0
            if __INLINED(moist):
                flux_sqv = 0.0
                flux_sqc = 0.0
                flux_sqr = 0.0


def get_upwind_flux(w, phi):
    nx, ny, nz = phi.shape

    f = deepcopy(phi)

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(1, nz):
                f[i, j, k] = w[i, j, k] * (
                    phi[i, j, k] if w[i, j, k] > 0 else phi[i, j, k - 1]
                )

    return f


def get_centered_flux(w, phi):
    f = deepcopy(phi)

    kstop = w.shape[2] - 1

    f[:, :, 1:kstop] = (
        w[:, :, 1:-1] * 0.5 * (phi[:, :, : kstop - 1] + phi[:, :, 1:kstop])
    )

    return f


def get_third_order_upwind_flux(w, phi):
    f4 = deepcopy(phi)

    kstop = w.shape[2] - 2

    f4[:, :, 2:kstop] = (
        w[:, :, 2:-2]
        / 12.0
        * (
            7.0 * (phi[:, :, 1 : kstop - 1] + phi[:, :, 2:kstop])
            - (phi[:, :, : kstop - 2] + phi[:, :, 3 : kstop + 1])
        )
    )

    f = deepcopy(phi)

    f[:, :, 2:kstop] = f4[:, :, 2:kstop] - np.abs(w[:, :, 2:-2]) / 12.0 * (
        3.0 * (phi[:, :, 1 : kstop - 1] - phi[:, :, 2:kstop])
        - (phi[:, :, : kstop - 2] - phi[:, :, 3 : kstop + 1])
    )

    return f


def get_fifth_order_upwind_flux(w, phi):
    f6 = deepcopy(phi)

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

    f = deepcopy(phi)

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


def validation_numpy(flux_scheme, domain, field, timestep, dtype):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx, ny, nz + 1)
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    w = zeros(storage_shape, False, dtype=dtype)
    w[...] = field[1:, 1:, :-1]
    s = zeros(storage_shape, False, dtype=dtype)
    s[...] = field[:-1, :-1, :-1]
    su = zeros(storage_shape, False, dtype=dtype)
    su[...] = field[1:, :-1, :-1]
    sv = zeros(storage_shape, False, dtype=dtype)
    sv[...] = field[:-1, :-1, :-1]
    sqv = zeros(storage_shape, False, dtype=dtype)
    sqv[...] = field[:-1, :-1, 1:]
    sqc = zeros(storage_shape, False, dtype=dtype)
    sqc[...] = field[1:, :-1, 1:]
    sqr = zeros(storage_shape, False, dtype=dtype)
    sqr[...] = field[:-1, :-1, 1:]

    #
    # dry
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, False, False)
    assert isinstance(core, flux_type)

    z = slice(nb, grid.nz - nb + 1)
    dz = grid.dz.to_units("K").values.item()

    fs, fsu, fsv = core.call_numpy(timestep, dz, w, s, su, sv)

    flux_s = get_fluxes(w, s)
    compare_arrays(fs, flux_s[:, :, z])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu, flux_su[:, :, z])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv, flux_sv[:, :, z])

    #
    # moist
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, True, False)
    assert isinstance(core, flux_type)

    fs, fsu, fsv, fsqv, fsqc, fsqr = core.call_numpy(
        timestep, dz, w, s, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
    )

    flux_s = get_fluxes(w, s)
    compare_arrays(fs, flux_s[:, :, z])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu, flux_su[:, :, z])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv, flux_sv[:, :, z])

    flux_sqv = get_fluxes(w, sqv)
    compare_arrays(fsqv, flux_sqv[:, :, z])

    flux_sqc = get_fluxes(w, sqc)
    compare_arrays(fsqc, flux_sqc[:, :, z])

    flux_sqr = get_fluxes(w, sqr)
    compare_arrays(fsqr, flux_sqr[:, :, z])


def validation_gt(
    flux_scheme, domain, field, timestep, backend, dtype, default_origin, rebuild
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx, ny, nz + 1)
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    w = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    w[...] = field[1:, 1:, :-1]
    s = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    s[...] = field[:-1, :-1, :-1]
    su = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    su[...] = field[1:, :-1, :-1]
    sv = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    sv[...] = field[:-1, :-1, :-1]
    sqv = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    sqv[...] = field[:-1, :-1, 1:]
    sqc = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    sqc[...] = field[1:, :-1, 1:]
    sqr = zeros(
        storage_shape, True, backend=backend, dtype=dtype, default_origin=default_origin
    )
    sqr[...] = field[:-1, :-1, 1:]

    #
    # dry
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, False, True)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, backend, dtype, default_origin, rebuild)

    z = slice(nb, grid.nz - nb + 1)
    dz = grid.dz.to_units("K").values.item()

    fs, fsu, fsv = ws(timestep, dz, w, s, su, sv)

    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, z], flux_s[:, :, z])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, z], flux_su[:, :, z])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, z], flux_sv[:, :, z])

    #
    # moist
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, True, True)
    assert isinstance(core, flux_type)
    ws = WrappingStencil(core, backend, dtype, default_origin, rebuild)

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
def test_upwind_numpy(data):
    # ========================================
    # random data generation
    # ========================================
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    domain = data.draw(
        st_domain(zaxis_length=(3, 40), gt_powered=False, dtype=dtype), label="domain"
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field((nx + 1, ny + 1, nz + 2), -1e4, 1e4, gt_powered=False, dtype=dtype)
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_numpy("upwind", domain, field, timestep, dtype)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_upwind_gt(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(3, 40), gt_powered=True, backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 2),
            -1e4,
            1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_gt(
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
def test_centered_numpy(data):
    # ========================================
    # random data generation
    # ========================================
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    domain = data.draw(
        st_domain(zaxis_length=(3, 40), gt_powered=False, dtype=dtype), label="domain"
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field((nx + 1, ny + 1, nz + 2), -1e4, 1e4, gt_powered=False, dtype=dtype)
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_numpy("centered", domain, field, timestep, dtype)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_centered_gt(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(3, 40), gt_powered=True, backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 2),
            -1e4,
            1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_gt(
        "centered",
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
def test_third_order_upwind_numpy(data):
    # ========================================
    # random data generation
    # ========================================
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    domain = data.draw(
        st_domain(zaxis_length=(5, 40), gt_powered=False, dtype=dtype), label="domain"
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field((nx + 1, ny + 1, nz + 2), -1e4, 1e4, gt_powered=False, dtype=dtype)
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_numpy("third_order_upwind", domain, field, timestep, dtype)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_third_order_upwind_gt(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(5, 40), gt_powered=True, backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 2),
            -1e4,
            1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_gt(
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
def test_fifth_order_upwind_numpy(data):
    # ========================================
    # random data generation
    # ========================================
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    domain = data.draw(
        st_domain(zaxis_length=(7, 40), gt_powered=False, dtype=dtype), label="domain"
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field((nx + 1, ny + 1, nz + 2), -1e4, 1e4, gt_powered=False, dtype=dtype)
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_numpy("fifth_order_upwind", domain, field, timestep, dtype)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_fifth_order_upwind_gt(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(7, 40), gt_powered=True, backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 2),
            -1e4,
            1e4,
            gt_powered=True,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    timestep = data.draw(st_floats(min_value=0, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    validation_gt(
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
