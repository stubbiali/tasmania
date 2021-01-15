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

from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.isentropic.dynamics.subclasses.minimal_vertical_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.backend import is_gt, get_gt_backend

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
    obj = IsentropicMinimalVerticalFlux.factory("upwind", backend="numpy")
    assert isinstance(obj, Upwind)
    obj = IsentropicMinimalVerticalFlux.factory("centered", backend="numpy")
    assert isinstance(obj, Centered)
    obj = IsentropicMinimalVerticalFlux.factory(
        "third_order_upwind", backend="numpy"
    )
    assert isinstance(obj, ThirdOrderUpwind)
    obj = IsentropicMinimalVerticalFlux.factory(
        "fifth_order_upwind", backend="numpy"
    )
    assert isinstance(obj, FifthOrderUpwind)


class WrappingStencil(StencilFactory):
    def __init__(self, cls, scheme, backend, backend_options, storage_options):
        super().__init__(backend, backend_options, storage_options)
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

    def call_dry(self, dt, dz, w, s, su, sv):
        mi, mj, mk = s.shape
        nb = self.core.extent

        stencil_args = {
            "w": w,
            "s": s,
            "su": su,
            "sv": sv,
            "flux_s": self.zeros(shape=(mi, mj, mk)),
            "flux_su": self.zeros(shape=(mi, mj, mk)),
            "flux_sv": self.zeros(shape=(mi, mj, mk)),
        }

        self.stencil_dry(
            **stencil_args,
            dt=dt,
            dz=dz,
            origin=(0, 0, nb),
            domain=(mi, mj, mk - 2 * nb)
        )

        return_list_names = ["flux_s", "flux_su", "flux_sv"]
        return_list = tuple(stencil_args[name] for name in return_list_names)

        return return_list

    def call_moist(
        self, dt, dz, w, sqv, sqc, sqr,
    ):
        mi, mj, mk = sqv.shape
        nb = self.core.extent

        stencil_args = {
            "w": w,
            "sqv": sqv,
            "sqc": sqc,
            "sqr": sqr,
            "flux_sqv": self.zeros(shape=(mi, mj, mk)),
            "flux_sqc": self.zeros(shape=(mi, mj, mk)),
            "flux_sqr": self.zeros(shape=(mi, mj, mk)),
        }

        self.stencil_moist(
            **stencil_args,
            dt=dt,
            dz=dz,
            origin=(0, 0, nb),
            domain=(mi, mj, mk - 2 * nb)
        )

        return_list_names = ["flux_sqv", "flux_sqc", "flux_sqr"]
        return_list = tuple(stencil_args[name] for name in return_list_names)

        return return_list

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil_dry")
    def stencil_dry_numpy(
        self,
        w,
        s,
        su,
        sv,
        flux_s,
        flux_su,
        flux_sv,
        *,
        dt=0.0,
        dz=0.0,
        origin,
        domain,
        **kwargs
    ):
        ijk = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        ijk_ext = (
            ijk[0],
            ijk[1],
            slice(
                origin[2] - self.core.extent,
                origin[2] + domain[2] + self.core.extent,
            ),
        )

        (
            flux_s[ijk],
            flux_su[ijk],
            flux_sv[ijk],
        ) = self.core.stencil_subroutine("flux_dry")(
            dt, dz, w[ijk_ext], s[ijk_ext], su[ijk_ext], sv[ijk_ext],
        )

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil_moist")
    def stencil_moist_numpy(
        self,
        w,
        sqv,
        sqc,
        sqr,
        flux_sqv,
        flux_sqc,
        flux_sqr,
        *,
        dt=0.0,
        dz=0.0,
        origin,
        domain,
        **kwargs
    ):
        ijk = tuple(slice(o, o + d) for o, d in zip(origin, domain))
        ijk_ext = (
            ijk[0],
            ijk[1],
            slice(
                origin[2] - self.core.extent,
                origin[2] + domain[2] + self.core.extent,
            ),
        )

        (
            flux_sqv[ijk],
            flux_sqc[ijk],
            flux_sqr[ijk],
        ) = self.core.stencil_subroutine("flux_moist")(
            dt, dz, w[ijk_ext], sqv[ijk_ext], sqc[ijk_ext], sqr[ijk_ext],
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil_dry")
    def stencil_dry_gt4py(
        w: gtscript.Field["dtype"],
        s: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        flux_s: gtscript.Field["dtype"],
        flux_su: gtscript.Field["dtype"],
        flux_sv: gtscript.Field["dtype"],
        *,
        dt: float = 0.0,
        dz: float = 0.0
    ):
        from __externals__ import get_flux_dry

        with computation(PARALLEL), interval(...):
            flux_s, flux_su, flux_sv = get_flux_dry(
                dt=dt, dz=dz, w=w, s=s, su=su, sv=sv
            )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil_moist")
    def stencil_moist_gt4py(
        w: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        flux_sqv: gtscript.Field["dtype"],
        flux_sqc: gtscript.Field["dtype"],
        flux_sqr: gtscript.Field["dtype"],
        *,
        dt: float = 0.0,
        dz: float = 0.0
    ):
        from __externals__ import get_flux_moist

        with computation(PARALLEL), interval(...):
            flux_sqv, flux_sqc, flux_sqr = get_flux_moist(
                dt=dt, dz=dz, w=w, sqv=sqv, sqc=sqc, sqr=sqr
            )


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
    storage_shape = (nx, ny, nz + 1)
    flux_type = flux_properties[flux_scheme]["type"]
    nb = flux_type.extent
    get_fluxes = flux_properties[flux_scheme]["get_fluxes"]

    dz = grid.dz.to_units("K").values.item()

    ws = WrappingStencil(
        cls, flux_scheme, backend, backend_options, storage_options
    )

    w = ws.zeros(shape=storage_shape)
    w[...] = field[1:, 1:, :-1]
    s = ws.zeros(shape=storage_shape)
    s[...] = field[:-1, :-1, :-1]
    su = ws.zeros(shape=storage_shape)
    su[...] = field[1:, :-1, :-1]
    sv = ws.zeros(shape=storage_shape)
    sv[...] = field[:-1, :-1, :-1]
    sqv = ws.zeros(shape=storage_shape)
    sqv[...] = field[:-1, :-1, 1:]
    sqc = ws.zeros(shape=storage_shape)
    sqc[...] = field[1:, :-1, 1:]
    sqr = ws.zeros(shape=storage_shape)
    sqr[...] = field[:-1, :-1, 1:]

    #
    # dry
    #
    fs, fsu, fsv = ws.call_dry(dt, dz, w, s, su, sv)
    z = slice(nb, grid.nz - nb + 1)
    flux_s = get_fluxes(w, s)
    compare_arrays(fs[:, :, z], flux_s[:, :, z])
    flux_su = get_fluxes(w, su)
    compare_arrays(fsu[:, :, z], flux_su[:, :, z])
    flux_sv = get_fluxes(w, sv)
    compare_arrays(fsv[:, :, z], flux_sv[:, :, z])

    #
    # moist
    #
    fsqv, fsqc, fsqr = ws.call_moist(dt, dz, w, sqv, sqc, sqr)
    flux_sqv = get_fluxes(w, sqv)
    compare_arrays(fsqv[:, :, z], flux_sqv[:, :, z])
    flux_sqc = get_fluxes(w, sqc)
    compare_arrays(fsqc[:, :, z], flux_sqc[:, :, z])
    flux_sqr = get_fluxes(w, sqr)
    compare_arrays(fsqr[:, :, z], flux_sqr[:, :, z])


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
        st_domain(zaxis_length=(4, 40), backend=backend, dtype=dtype),
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

    dt = data.draw(st_floats(min_value=0, max_value=3600), label="dt")

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)
    validation(
        IsentropicMinimalVerticalFlux,
        scheme,
        domain,
        field,
        dt,
        backend,
        bo,
        so,
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_numerics("third_order_upwind", "gt4py:numpy", float)
