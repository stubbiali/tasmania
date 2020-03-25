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
import pytest

import gt4py as gt

from tasmania.python.isentropic.physics.vertical_advection import (
    IsentropicVerticalAdvection,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

from tests.conf import default_origin as conf_dorigin, datatype as conf_dtype
from tests.isentropic.test_isentropic_vertical_advection import validation
from tests.utilities import st_domain, st_isentropic_state_f, st_one_of, st_raw_field


class DebugIsentropicVerticalAdvection(IsentropicVerticalAdvection):
    @staticmethod
    def _stencil_gt_defs(
        in_w: gt.gtscript.Field["dtype"],
        in_s: gt.gtscript.Field["dtype"],
        in_su: gt.gtscript.Field["dtype"],
        in_sv: gt.gtscript.Field["dtype"],
        out_s: gt.gtscript.Field["dtype"],
        out_su: gt.gtscript.Field["dtype"],
        out_sv: gt.gtscript.Field["dtype"],
        in_qv: gt.gtscript.Field["dtype"] = None,
        in_qc: gt.gtscript.Field["dtype"] = None,
        in_qr: gt.gtscript.Field["dtype"] = None,
        out_qv: gt.gtscript.Field["dtype"] = None,
        out_qc: gt.gtscript.Field["dtype"] = None,
        out_qr: gt.gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dz: float
    ) -> None:
        from __externals__ import moist, vflux, vflux_end, vflux_extent, vstaggering

        # interpolate the velocity on the interface levels
        with computation(PARALLEL), interval(0, 1):
            w = 0.0
        with computation(PARALLEL), interval(1, None):
            if __INLINED(vstaggering):  # compile-time if
                w = in_w
            else:
                w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, -1])

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if __INLINED(vstaggering):
                wc = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                wc = in_w

        # compute the isentropic_prognostic density of the water species
        if __INLINED(moist):  # compile-time if
            with computation(PARALLEL), interval(0, None):
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr

        # compute the fluxes
        with computation(PARALLEL), interval(vflux_extent, vflux_end):
            if __INLINED(not moist):  # compile-time if
                flux_s, flux_su, flux_sv = vflux(
                    dt=dt, dz=dz, w=w, s=in_s, su=in_su, sv=in_sv
                )
            else:
                flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = vflux(
                    dt=dt,
                    dz=dz,
                    w=w,
                    s=in_s,
                    su=in_su,
                    sv=in_sv,
                    sqv=sqv,
                    sqc=sqc,
                    sqr=sqr,
                )

        # calculate the tendencies
        with computation(PARALLEL), interval(0, vflux_extent):
            out_s = 0.0
            out_su = 0.0
            out_sv = 0.0
            if __INLINED(moist):  # compile-time if
                out_qv = 0.0
                out_qc = 0.0
                out_qr = 0.0
        with computation(PARALLEL), interval(vflux_extent, -vflux_extent):
            out_s = (flux_s[0, 0, 1] - flux_s[0, 0, 0]) / dz
            out_su = (flux_su[0, 0, 1] - flux_su[0, 0, 0]) / dz
            out_sv = (flux_sv[0, 0, 1] - flux_sv[0, 0, 0]) / dz
            if __INLINED(moist):  # compile-time if
                out_qv = (flux_sqv[0, 0, 1] - flux_sqv[0, 0, 0]) / (in_s[0, 0, 0] * dz)
                out_qc = (flux_sqc[0, 0, 1] - flux_sqc[0, 0, 0]) / (in_s[0, 0, 0] * dz)
                out_qr = (flux_sqr[0, 0, 1] - flux_sqr[0, 0, 0]) / (in_s[0, 0, 0] * dz)
        with computation(PARALLEL), interval(-vflux_extent, None):
            out_s = 0.0
            out_su = 0.0
            out_sv = 0.0
            if __INLINED(moist):  # compile-time if
                out_qv = 0.0
                out_qc = 0.0
                out_qr = 0.0


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_upwind(data, subtests):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = True
    backend = "gtmc"
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(zaxis_length=(3, 20), gt_powered=True, backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        domain,
        "upwind",
        False,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "upwind",
        False,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "upwind",
        True,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "upwind",
        True,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_centered(data, subtests):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = True
    backend = "gtmc"
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            zaxis_length=(3, 20), gt_powered=gt_powered, backend=backend, dtype=dtype
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        domain,
        "centered",
        False,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "centered",
        False,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "centered",
        True,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "centered",
        True,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_third_order_upwind(data, subtests):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = True
    backend = "gtmc"
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            zaxis_length=(5, 20), gt_powered=gt_powered, backend=backend, dtype=dtype
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        domain,
        "third_order_upwind",
        False,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "third_order_upwind",
        False,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "third_order_upwind",
        True,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )
    validation(
        domain,
        "third_order_upwind",
        True,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        cls=DebugIsentropicVerticalAdvection,
        subtests=subtests
    )


if __name__ == "__main__":
    pytest.main([__file__])
