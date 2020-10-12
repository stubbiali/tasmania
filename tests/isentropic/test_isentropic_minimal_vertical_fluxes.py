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

from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.isentropic.dynamics.subclasses.minimal_vertical_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


def test_registry():
    assert "upwind" in IsentropicMinimalVerticalFlux.registry
    assert IsentropicMinimalVerticalFlux.registry["upwind"] == Upwind
    assert "centered" in IsentropicMinimalVerticalFlux.registry
    assert IsentropicMinimalVerticalFlux.registry["centered"] == Centered
    assert "third_order_upwind" in IsentropicMinimalVerticalFlux.registry
    assert (
        IsentropicMinimalVerticalFlux.registry["third_order_upwind"]
        == ThirdOrderUpwind
    )
    assert "fifth_order_upwind" in IsentropicMinimalVerticalFlux.registry
    assert (
        IsentropicMinimalVerticalFlux.registry["fifth_order_upwind"]
        == FifthOrderUpwind
    )


def test_factory():
    obj = IsentropicMinimalVerticalFlux.factory("upwind", False, "numpy")
    assert isinstance(obj, Upwind)
    obj = IsentropicMinimalVerticalFlux.factory("centered", False, "numpy")
    assert isinstance(obj, Centered)
    obj = IsentropicMinimalVerticalFlux.factory(
        "third_order_upwind", False, "numpy"
    )
    assert isinstance(obj, ThirdOrderUpwind)
    obj = IsentropicMinimalVerticalFlux.factory(
        "fifth_order_upwind", False, "numpy"
    )
    assert isinstance(obj, FifthOrderUpwind)


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
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_su": zeros(
                storage_shape,
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_sv": zeros(
                storage_shape,
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
                        backend=self.backend,
                        dtype=s.dtype,
                        default_origin=self.default_origin,
                    ),
                    "flux_sqc": zeros(
                        storage_shape,
                        backend=self.backend,
                        dtype=s.dtype,
                        default_origin=self.default_origin,
                    ),
                    "flux_sqr": zeros(
                        storage_shape,
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
            get_gt_backend(self.backend),
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
                flux_s, flux_su, flux_sv = core(
                    dt=dt, dz=dz, w=w, s=s, su=su, sv=sv
                )
            else:
                flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = core(
                    dt=dt,
                    dz=dz,
                    w=w,
                    s=s,
                    su=su,
                    sv=sv,
                    sqv=sqv,
                    sqc=sqc,
                    sqr=sqr,
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


def validation(
    flux_scheme,
    domain,
    field,
    timestep,
    backend,
    dtype,
    default_origin,
    rebuild,
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx, ny, nz + 1)
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    w = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    w[...] = field[1:, 1:, :-1]
    s = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    s[...] = field[:-1, :-1, :-1]
    su = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    su[...] = field[1:, :-1, :-1]
    sv = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sv[...] = field[:-1, :-1, :-1]
    sqv = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sqv[...] = field[:-1, :-1, 1:]
    sqc = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sqc[...] = field[1:, :-1, 1:]
    sqr = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    sqr[...] = field[:-1, :-1, 1:]

    #
    # dry
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, False, backend)
    assert isinstance(core, flux_type)

    zl = slice(nb, grid.nz - nb + 1) if is_gt(backend) else slice(0, None)
    zr = slice(nb, grid.nz - nb + 1)
    dz = grid.dz.to_units("K").values.item()

    if is_gt(backend):
        ws = WrappingStencil(core, backend, dtype, default_origin, rebuild)
        fs, fsu, fsv = ws(timestep, dz, w, s, su, sv)
    else:
        fs, fsu, fsv = core.call_numpy(timestep, dz, w, s, su, sv)

    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, zl], flux_s[:, :, zr])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, zl], flux_su[:, :, zr])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, zl], flux_sv[:, :, zr])

    #
    # moist
    #
    core = IsentropicMinimalVerticalFlux.factory(flux_scheme, True, backend)
    assert isinstance(core, flux_type)

    if is_gt(backend):
        ws = WrappingStencil(core, backend, dtype, default_origin, rebuild)
        fs, fsu, fsv, fsqv, fsqc, fsqr = ws(
            timestep, dz, w, s, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
        )
    else:
        fs, fsu, fsv, fsqv, fsqc, fsqr = core.call_numpy(
            timestep, dz, w, s, su, sv, sqv=sqv, sqc=sqc, sqr=sqr
        )

    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, zl], flux_s[:, :, zr])

    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, zl], flux_su[:, :, zr])

    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, zl], flux_sv[:, :, zr])

    flux_sqv = get_fluxes(w, sqv)
    compare_arrays(fsqv[:, :, zl], flux_sqv[:, :, zr])

    flux_sqc = get_fluxes(w, sqc)
    compare_arrays(fsqc[:, :, zl], flux_sqc[:, :, zr])

    flux_sqr = get_fluxes(w, sqr)
    compare_arrays(fsqr[:, :, zl], flux_sqr[:, :, zr])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "scheme",
    tuple(
        scheme
        for scheme in flux_properties.keys()
        if scheme != "fifth_order_upwind"
    ),
)
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_numerics(data, scheme, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(3, 40), backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    field = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 2),
            -1e4,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
    )

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        scheme,
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
