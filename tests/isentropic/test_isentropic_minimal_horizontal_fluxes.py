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
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from gt4py import gtscript

from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.subclasses.minimal_horizontal_fluxes import (
    Upwind,
    Centered,
    ThirdOrderUpwind,
    FifthOrderUpwind,
)
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import get_gt_backend

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.isentropic.test_isentropic_horizontal_fluxes import (
    get_centered_fluxes,
    get_fifth_order_upwind_fluxes,
    get_third_order_upwind_fluxes,
    get_upwind_fluxes,
    validation,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_raw_field
from tests.utilities import hyp_settings


def test_registry():
    assert "upwind" in IsentropicMinimalHorizontalFlux.registry
    assert IsentropicMinimalHorizontalFlux.registry["upwind"] == Upwind
    assert "centered" in IsentropicMinimalHorizontalFlux.registry
    assert IsentropicMinimalHorizontalFlux.registry["centered"] == Centered
    assert "third_order_upwind" in IsentropicMinimalHorizontalFlux.registry
    assert (
        IsentropicMinimalHorizontalFlux.registry["third_order_upwind"]
        == ThirdOrderUpwind
    )
    assert "fifth_order_upwind" in IsentropicMinimalHorizontalFlux.registry
    assert (
        IsentropicMinimalHorizontalFlux.registry["fifth_order_upwind"]
        == FifthOrderUpwind
    )


def test_factory():
    obj = IsentropicMinimalHorizontalFlux.factory("upwind", False, "numpy")
    assert isinstance(obj, Upwind)
    obj = IsentropicMinimalHorizontalFlux.factory("centered", False, "numpy")
    assert isinstance(obj, Centered)
    obj = IsentropicMinimalHorizontalFlux.factory(
        "third_order_upwind", False, "numpy"
    )
    assert isinstance(obj, ThirdOrderUpwind)
    obj = IsentropicMinimalHorizontalFlux.factory(
        "fifth_order_upwind", False, "numpy"
    )
    assert isinstance(obj, FifthOrderUpwind)


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
            "flux_s_x": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_s_y": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_su_x": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_su_y": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_sv_x": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
            "flux_sv_y": zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            ),
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

        moist = self.core.moist
        if moist:
            stencil_args["sqv"] = sqv
            stencil_args["flux_sqv_x"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            )
            stencil_args["flux_sqv_y"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            )
            stencil_args["sqc"] = sqc
            stencil_args["flux_sqc_x"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            )
            stencil_args["flux_sqc_y"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            )
            stencil_args["sqr"] = sqr
            stencil_args["flux_sqr_x"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
            )
            stencil_args["flux_sqr_y"] = zeros(
                (mi, mj, mk),
                backend=self.backend,
                dtype=s.dtype,
                default_origin=self.default_origin,
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
            "core": self.core.call,
            "moist": moist,
            "s_tnd_on": s_tnd_on,
            "su_tnd_on": su_tnd_on,
            "sv_tnd_on": sv_tnd_on,
        }
        if moist:
            externals.update(
                {
                    "qv_tnd_on": qv_tnd_on,
                    "qc_tnd_on": qc_tnd_on,
                    "qr_tnd_on": qr_tnd_on,
                }
            )

        decorator = gtscript.stencil(
            get_gt_backend(self.backend),
            dtypes={"dtype": self.dtype},
            externals=externals,
            rebuild=self.rebuild,
        )
        stencil = decorator(self.stencil_defs)

        nb = self.nb
        stencil(
            **stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb, nb, 0)},
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
        sqv: gtscript.Field["dtype"] = None,
        sqc: gtscript.Field["dtype"] = None,
        sqr: gtscript.Field["dtype"] = None,
        flux_sqv_x: gtscript.Field["dtype"] = None,
        flux_sqv_y: gtscript.Field["dtype"] = None,
        flux_sqc_x: gtscript.Field["dtype"] = None,
        flux_sqc_y: gtscript.Field["dtype"] = None,
        flux_sqr_x: gtscript.Field["dtype"] = None,
        flux_sqr_y: gtscript.Field["dtype"] = None,
        s_tnd: gtscript.Field["dtype"] = None,
        su_tnd: gtscript.Field["dtype"] = None,
        sv_tnd: gtscript.Field["dtype"] = None,
        qv_tnd: gtscript.Field["dtype"] = None,
        qc_tnd: gtscript.Field["dtype"] = None,
        qr_tnd: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0
    ):
        from __externals__ import core, moist

        with computation(PARALLEL), interval(...):
            if __INLINED(not moist):
                (
                    flux_s_x,
                    flux_s_y,
                    flux_su_x,
                    flux_su_y,
                    flux_sv_x,
                    flux_sv_y,
                ) = core(
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
                (
                    flux_s_x,
                    flux_s_y,
                    flux_su_x,
                    flux_su_y,
                    flux_sv_x,
                    flux_sv_y,
                    flux_sqv_x,
                    flux_sqv_y,
                    flux_sqc_x,
                    flux_sqc_y,
                    flux_sqr_x,
                    flux_sqr_y,
                ) = core(
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

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        "upwind",
        domain,
        field,
        timestep,
        backend,
        dtype,
        default_origin,
        rebuild=False,
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

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        "centered",
        domain,
        field,
        timestep,
        backend,
        dtype,
        default_origin,
        rebuild=False,
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

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

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

    timestep = data.draw(
        st_floats(min_value=0, max_value=3600), label="timestep"
    )

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
