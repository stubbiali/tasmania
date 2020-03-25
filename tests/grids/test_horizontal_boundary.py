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
    assume,
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import numpy as np
import pytest

from tasmania.python.grids.grid import NumericalGrid
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids._horizontal_boundary import (
    Relaxed,
    Relaxed1DX,
    Relaxed1DY,
    Periodic,
    Periodic1DX,
    Periodic1DY,
    Dirichlet,
    Dirichlet1DX,
    Dirichlet1DY,
    Identity,
    Identity1DX,
    Identity1DY,
)
from tasmania.python.utils.storage_utils import get_numerical_state
from tasmania.python.utils.utils import equal_to as eq

from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    st_burgers_state,
    st_horizontal_boundary_kwargs,
    st_horizontal_boundary_layers,
    st_isentropic_state,
    st_physical_grid,
)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def test_relaxed(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )
    hb = HorizontalBoundary.factory("relaxed", nx, ny, nb, **hb_kwargs)

    state = data.draw(
        st_isentropic_state(grid, moist=True, precipitation=True), label="state"
    )

    cgrid = NumericalGrid(grid, hb)
    cstate = data.draw(
        st_isentropic_state(cgrid, moist=True, precipitation=True), label="ref_state"
    )

    # ========================================
    # test
    # ========================================
    if ny == 1:
        assert isinstance(hb, Relaxed1DX)
    elif nx == 1:
        assert isinstance(hb, Relaxed1DY)
    else:
        assert isinstance(hb, Relaxed)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb

    if ny == 1:
        assert hb.ni == nx
        assert hb.nj == 2 * nb + 1
    elif nx == 1:
        assert hb.ni == 2 * nb + 1
        assert hb.nj == ny
    else:
        assert hb.ni == nx
        assert hb.nj == ny

    assert hb.type == "relaxed"

    assert "nr" in hb.kwargs
    assert hb.kwargs["nr"] == hb_kwargs["nr"]
    assert len(hb.kwargs) == 1

    #
    # x-axis
    #
    x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

    if nx == 1:
        cx = hb.get_numerical_xaxis(x)
        assert len(cx.dims) == 1
        assert cx.dims[0] == x.dims[0]
        assert cx.attrs["units"] == x.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cx.values[i], x.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cx.values[nb] == x.values[0]
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], x.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == x.values[0]

        cx = hb.get_numerical_xaxis(xu)
        assert len(cx.dims) == 1
        assert cx.dims[0] == xu.dims[0]
        assert cx.attrs["units"] == xu.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 2
        assert all([eq(cx.values[i], xu.values[0] - nb + i) for i in range(nb)])
        assert cx.values[nb] == xu.values[0]
        assert cx.values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == xu.values[0]
        assert cx.coords[cx.dims[0]].values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(x, hb.get_numerical_xaxis(x))
        compare_dataarrays(xu, hb.get_numerical_xaxis(xu))

    compare_dataarrays(x, hb.get_physical_xaxis(hb.get_numerical_xaxis(x)))
    compare_dataarrays(xu, hb.get_physical_xaxis(hb.get_numerical_xaxis(xu)))

    #
    # y-axis
    #
    y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

    if ny == 1:
        cy = hb.get_numerical_yaxis(y)
        assert len(cy.dims) == 1
        assert cy.dims[0] == y.dims[0]
        assert cy.attrs["units"] == y.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cy.values[i], y.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cy.values[nb] == y.values[0]
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], y.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == y.values[0]

        cy = hb.get_numerical_yaxis(yv)
        assert len(cy.dims) == 1
        assert cy.dims[0] == yv.dims[0]
        assert cy.attrs["units"] == yv.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 2
        assert all([eq(cy.values[i], yv.values[0] - nb + i) for i in range(nb)])
        assert cy.values[nb] == yv.values[0]
        assert cy.values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == yv.values[0]
        assert cy.coords[cy.dims[0]].values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(y, hb.get_numerical_yaxis(y))
        compare_dataarrays(yv, hb.get_numerical_yaxis(yv))

    compare_dataarrays(y, hb.get_physical_yaxis(hb.get_numerical_yaxis(y)))
    compare_dataarrays(yv, hb.get_physical_yaxis(hb.get_numerical_yaxis(yv)))

    #
    # numerical and physical field
    #
    field = state["air_isentropic_density"].values
    field_stgx = state["x_velocity_at_u_locations"].values
    field_stgy = state["y_velocity_at_v_locations"].values

    if ny == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=1)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx), np.repeat(field_stgx, 2 * nb + 1, axis=1)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy)[:, : nb + 1, :],
            np.repeat(field_stgy[:, 0:1, :], nb + 1, axis=1),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy)[:, nb + 1 :, :],
            np.repeat(field_stgy[:, -1:, :], nb + 1, axis=1),
        )
    elif nx == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=0)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx)[: nb + 1, :, :],
            np.repeat(field_stgx[0:1, :, :], nb + 1, axis=0),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx)[nb + 1 :, :, :],
            np.repeat(field_stgx[-1:, :, :], nb + 1, axis=0),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy), np.repeat(field_stgy, 2 * nb + 1, axis=0)
        )
    else:
        compare_arrays(hb.get_numerical_field(field), field)
        compare_arrays(hb.get_numerical_field(field_stgx), field_stgx)
        compare_arrays(hb.get_numerical_field(field_stgy), field_stgy)

    compare_arrays(hb.get_physical_field(hb.get_numerical_field(field)), field)

    #
    # reference state
    #
    hb.reference_state = cstate

    for key in cstate:
        with subtests.test(key=key):
            if key != "time":
                compare_dataarrays(
                    cstate[key], hb.reference_state[key], compare_coordinate_values=False
                )

    #
    # enforce_field
    #
    field_names = (
        "air_isentropic_density",
        "x_velocity_at_u_locations",
        "y_velocity_at_v_locations",
        "air_pressure_on_interface_levels",
        "precipitation",
    )
    for name in field_names:
        field = cstate[name].values
        units = cstate[name].attrs["units"]
        hb.enforce_field(field, name, units, cstate["time"])

    #
    # enforce_raw
    #
    rcstate = {"time": cstate["time"]}
    for key in state:
        if key != "time":
            rcstate[key] = cstate[key].values

    hb.enforce_raw(rcstate)

    #
    # enforce
    #
    hb.enforce(cstate)

    #
    # set_outermost_layers_x
    #
    units = cstate["x_velocity_at_u_locations"].attrs["units"]
    hb.set_outermost_layers_x(
        cstate["x_velocity_at_u_locations"].values,
        field_name="x_velocity_at_u_locations",
        field_units=units,
    )
    compare_arrays(
        cstate["x_velocity_at_u_locations"].values[0, :],
        hb.reference_state["x_velocity_at_u_locations"].to_units(units).values[0, :],
    )
    compare_arrays(
        cstate["x_velocity_at_u_locations"].values[-1, :],
        hb.reference_state["x_velocity_at_u_locations"].to_units(units).values[-1, :],
    )

    #
    # set_outermost_layers_y
    #
    units = cstate["y_velocity_at_v_locations"].attrs["units"]
    hb.set_outermost_layers_y(
        cstate["y_velocity_at_v_locations"].values,
        field_name="y_velocity_at_v_locations",
        field_units=units,
    )
    compare_arrays(
        cstate["y_velocity_at_v_locations"].values[:, 0],
        hb.reference_state["y_velocity_at_v_locations"].to_units(units).values[:, 0],
    )
    compare_arrays(
        cstate["y_velocity_at_v_locations"].values[:, -1],
        hb.reference_state["y_velocity_at_v_locations"].to_units(units).values[:, -1],
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_periodic(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("periodic", nx, ny, nb), label="hb_kwargs"
    )

    state = data.draw(
        st_isentropic_state(grid, moist=True, precipitation=True), label="state"
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("periodic", nx, ny, nb, **hb_kwargs)

    if ny == 1:
        assert isinstance(hb, Periodic1DX)
    elif nx == 1:
        assert isinstance(hb, Periodic1DY)
    else:
        assert isinstance(hb, Periodic)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == nx + 2 * nb
    assert hb.nj == ny + 2 * nb

    assert hb.type == "periodic"

    assert len(hb.kwargs) == 0

    #
    # x-axis
    #
    x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

    if nx == 1:
        cx = hb.get_numerical_xaxis(x)
        assert len(cx.dims) == 1
        assert cx.dims[0] == x.dims[0]
        assert cx.attrs["units"] == x.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cx.values[i], x.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cx.values[nb] == x.values[0]
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], x.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == x.values[0]

        cx = hb.get_numerical_xaxis(xu)
        assert len(cx.dims) == 1
        assert cx.dims[0] == xu.dims[0]
        assert cx.attrs["units"] == xu.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 2
        assert all([eq(cx.values[i], xu.values[0] - nb + i) for i in range(nb)])
        assert cx.values[nb] == xu.values[0]
        assert cx.values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == xu.values[0]
        assert cx.coords[cx.dims[0]].values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        dx = grid.grid_xy.dx.values.item()

        cx = hb.get_numerical_xaxis(x)
        assert len(cx.dims) == 1
        assert cx.dims[0] == x.dims[0]
        assert cx.attrs["units"] == x.attrs["units"]
        assert cx.values.shape[0] == nx + 2 * nb
        assert all(
            [eq(cx.values[i + 1] - cx.values[i], dx) for i in range(nx + 2 * nb - 1)]
        )
        assert len(cx.coords) == 1
        assert all(
            [
                eq(
                    cx.coords[cx.dims[0]].values[i + 1] - cx.coords[cx.dims[0]].values[i],
                    dx,
                )
                for i in range(nx + 2 * nb - 1)
            ]
        )

        cx = hb.get_numerical_xaxis(xu)
        assert len(cx.dims) == 1
        assert cx.dims[0] == xu.dims[0]
        assert cx.attrs["units"] == xu.attrs["units"]
        assert cx.values.shape[0] == nx + 2 * nb + 1
        assert all([eq(cx.values[i + 1] - cx.values[i], dx) for i in range(nx + 2 * nb)])
        assert len(cx.coords) == 1
        assert all(
            [
                eq(
                    cx.coords[cx.dims[0]].values[i + 1] - cx.coords[cx.dims[0]].values[i],
                    dx,
                )
                for i in range(nx + 2 * nb)
            ]
        )

    #
    # y-axis
    #
    y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

    if ny == 1:
        cy = hb.get_numerical_yaxis(y)
        assert len(cy.dims) == 1
        assert cy.dims[0] == y.dims[0]
        assert cy.attrs["units"] == y.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cy.values[i], y.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cy.values[nb] == y.values[0]
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], y.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == y.values[0]

        cy = hb.get_numerical_yaxis(yv)
        assert len(cy.dims) == 1
        assert cy.dims[0] == yv.dims[0]
        assert cy.attrs["units"] == yv.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 2
        assert all([eq(cy.values[i], yv.values[0] - nb + i) for i in range(nb)])
        assert cy.values[nb] == yv.values[0]
        assert cy.values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == yv.values[0]
        assert cy.coords[cy.dims[0]].values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        dy = grid.grid_xy.dy.values.item()

        cy = hb.get_numerical_yaxis(y)
        assert len(cy.dims) == 1
        assert cy.dims[0] == y.dims[0]
        assert cy.attrs["units"] == y.attrs["units"]
        assert cy.values.shape[0] == ny + 2 * nb
        assert all(
            [eq(cy.values[i + 1] - cy.values[i], dy) for i in range(ny + 2 * nb - 1)]
        )
        assert len(cy.coords) == 1
        assert all(
            [
                eq(
                    cy.coords[cy.dims[0]].values[i + 1] - cy.coords[cy.dims[0]].values[i],
                    dy,
                )
                for i in range(ny + 2 * nb - 1)
            ]
        )

        cy = hb.get_numerical_yaxis(yv)
        assert len(cy.dims) == 1
        assert cy.dims[0] == yv.dims[0]
        assert cy.attrs["units"] == yv.attrs["units"]
        assert cy.values.shape[0] == ny + 2 * nb + 1
        assert all([eq(cy.values[i + 1] - cy.values[i], dy) for i in range(ny + 2 * nb)])
        assert len(cy.coords) == 1
        assert all(
            [
                eq(
                    cy.coords[cy.dims[0]].values[i + 1] - cy.coords[cy.dims[0]].values[i],
                    dy,
                )
                for i in range(ny + 2 * nb)
            ]
        )

    compare_dataarrays(x, hb.get_physical_xaxis(hb.get_numerical_xaxis(x)))
    compare_dataarrays(xu, hb.get_physical_xaxis(hb.get_numerical_xaxis(xu)))

    #
    # numerical and physical unstaggered field
    #
    field = state["air_isentropic_density"].values

    field[-1, :] = field[0, :]
    field[:, -1] = field[:, 0]
    cfield = hb.get_numerical_field(field)

    assert cfield.shape[0:2] == (nx + 2 * nb, ny + 2 * nb)

    avgx = (
        0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 : -nb + 1, :])
        if nb > 1
        else 0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 :, :])
    )
    compare_arrays(avgx[0, :], avgx[-1, :])
    if nx == 1:
        compare_arrays(avgx, cfield[nb:nb, :])
    avgy = (
        0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 : -nb + 1])
        if nb > 1
        else 0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 :])
    )
    compare_arrays(avgy[:, 0], avgy[:, -1])
    if ny == 1:
        compare_arrays(avgy, cfield[:, nb:nb])

    if nb > 1:
        avgx = (
            0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 : -nb + 2, :])
            if nb > 2
            else 0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-1, :])
        if nx == 1:
            compare_arrays(avgx, cfield[nb:nb, :])
        avgy = (
            0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 : -nb + 2])
            if nb > 2
            else 0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -1])
        if ny == 1:
            compare_arrays(avgy, cfield[:, nb:nb])

    if nb > 2:
        avgx = (
            0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 : -nb + 3, :])
            if nb > 3
            else 0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-1, :])
        if nx == 1:
            compare_arrays(avgx, cfield[nb:nb, :])
        avgy = (
            0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 : -nb + 3])
            if nb > 3
            else 0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -1])
        if ny == 1:
            compare_arrays(avgy, cfield[:, nb:nb])

    #
    # numerical and physical x-staggered field
    #
    field = state["x_velocity_at_u_locations"].values

    field[-2, :] = field[0, :]
    field[-1, :] = field[1, :]
    field[:, -1] = field[:, 0]
    cfield = hb.get_numerical_field(field)

    assert cfield.shape[0:2] == (nx + 2 * nb + 1, ny + 2 * nb)

    avgx = (
        0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 : -nb + 1, :])
        if nb > 1
        else 0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 :, :])
    )
    compare_arrays(avgx[0, :], avgx[-2, :])
    compare_arrays(avgx[1, :], avgx[-1, :])
    avgy = (
        0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 : -nb + 1])
        if nb > 1
        else 0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 :])
    )
    compare_arrays(avgy[:, 0], avgy[:, -1])
    if ny == 1:
        compare_arrays(avgy, cfield[:, nb:nb])

    if nb > 1:
        avgx = (
            0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 : -nb + 2, :])
            if nb > 2
            else 0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-2, :])
        compare_arrays(avgx[1, :], avgx[-1, :])
        avgy = (
            0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 : -nb + 2])
            if nb > 2
            else 0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -1])
        if ny == 1:
            compare_arrays(avgy, cfield[:, nb:nb])

    if nb > 2:
        avgx = (
            0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 : -nb + 3, :])
            if nb > 3
            else 0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-2, :])
        compare_arrays(avgx[1, :], avgx[-1, :])
        avgy = (
            0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 : -nb + 3])
            if nb > 3
            else 0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -1])
        if ny == 1:
            compare_arrays(avgy, cfield[:, nb:nb])

    #
    # numerical and physical y-staggered field
    #
    field = state["y_velocity_at_v_locations"].values

    field[-1, :] = field[0, :]
    field[:, -2] = field[:, 0]
    field[:, -1] = field[:, 1]
    cfield = hb.get_numerical_field(field)

    assert cfield.shape[0:2] == (nx + 2 * nb, ny + 2 * nb + 1)

    avgx = (
        0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 : -nb + 1, :])
        if nb > 1
        else 0.5 * (cfield[nb - 1 : -nb - 1, :] + cfield[nb + 1 :, :])
    )
    compare_arrays(avgx[0, :], avgx[-1, :])
    if nx == 1:
        compare_arrays(avgx, cfield[nb:nb, :])
    avgy = (
        0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 : -nb + 1])
        if nb > 1
        else 0.5 * (cfield[:, nb - 1 : -nb - 1] + cfield[:, nb + 1 :])
    )
    compare_arrays(avgy[:, 0], avgy[:, -2])
    compare_arrays(avgy[:, 1], avgy[:, -1])

    if nb > 1:
        avgx = (
            0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 : -nb + 2, :])
            if nb > 2
            else 0.5 * (cfield[nb - 2 : -nb - 2, :] + cfield[nb + 2 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-1, :])
        if nx == 1:
            compare_arrays(avgx, cfield[nb:nb, :])
        avgy = (
            0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 : -nb + 2])
            if nb > 2
            else 0.5 * (cfield[:, nb - 2 : -nb - 2] + cfield[:, nb + 2 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -2])
        compare_arrays(avgy[:, 1], avgy[:, -1])

    if nb > 2:
        avgx = (
            0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 : -nb + 3, :])
            if nb > 3
            else 0.5 * (cfield[nb - 3 : -nb - 3, :] + cfield[nb + 3 :, :])
        )
        compare_arrays(avgx[0, :], avgx[-1, :])
        if nx == 1:
            compare_arrays(avgx, cfield[nb:nb, :])
        avgy = (
            0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 : -nb + 3])
            if nb > 3
            else 0.5 * (cfield[:, nb - 3 : -nb - 3] + cfield[:, nb + 3 :])
        )
        compare_arrays(avgy[:, 0], avgy[:, -2])
        compare_arrays(avgy[:, 1], avgy[:, -1])

    #
    # enforce_field
    #
    field = state["air_isentropic_density"].values
    cfield = hb.get_numerical_field(field)
    vfield = deepcopy(cfield)
    hb.enforce_field(cfield)
    compare_arrays(cfield, vfield)

    field = state["x_velocity_at_u_locations"].values
    cfield = hb.get_numerical_field(field)
    vfield = deepcopy(cfield)
    hb.enforce_field(cfield)
    compare_arrays(cfield, vfield)

    field = state["y_velocity_at_v_locations"].values
    cfield = hb.get_numerical_field(field)
    vfield = deepcopy(cfield)
    hb.enforce_field(cfield)
    compare_arrays(cfield, vfield)

    #
    # enforce_raw
    #
    rcstate = {"time": state["time"]}
    for name in state:
        if name != "time":
            rcstate[name] = hb.get_numerical_field(state[name].values)

    hb.enforce_raw(rcstate)

    #
    # set_outermost_layers_x
    #
    pfield = state["x_velocity_at_u_locations"].values
    cfield = hb.get_numerical_field(pfield, "x_velocity_at_u_locations")
    hb.set_outermost_layers_x(cfield)
    compare_arrays(cfield[0, :], cfield[-2, :])
    compare_arrays(cfield[-1, :], cfield[1, :])

    #
    # set_outermost_layers_y
    #
    pfield = state["y_velocity_at_v_locations"].values
    cfield = hb.get_numerical_field(pfield, "y_velocity_at_v_locations")
    hb.set_outermost_layers_y(cfield)
    compare_arrays(cfield[:, 0], cfield[:, -2])
    compare_arrays(cfield[:, -1], cfield[:, 1])


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data, hyp_st.data())
def test_dirichlet_burgers(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_length=(1, 1)), label="grid")
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("dirichlet", nx, ny, nb), label="hb_kwargs"
    )
    hb = HorizontalBoundary.factory("dirichlet", nx, ny, nb, **hb_kwargs)

    state = data.draw(st_burgers_state(grid), label="state")

    cgrid = NumericalGrid(grid, hb)
    cstate = data.draw(st_burgers_state(cgrid), label="ref_state")

    # ========================================
    # test
    # ========================================
    if ny == 1:
        assert isinstance(hb, Dirichlet1DX)
    elif nx == 1:
        assert isinstance(hb, Dirichlet1DY)
    else:
        assert isinstance(hb, Dirichlet)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb

    if ny == 1:
        assert hb.ni == nx
        assert hb.nj == 2 * nb + 1
    elif nx == 1:
        assert hb.ni == 2 * nb + 1
        assert hb.nj == ny
    else:
        assert hb.ni == nx
        assert hb.nj == ny

    assert hb.type == "dirichlet"

    assert "core" in hb.kwargs
    assert hb.kwargs["core"] == hb_kwargs["core"]
    assert len(hb.kwargs) == 1

    #
    # x-axis
    #
    x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

    if nx == 1:
        cx = hb.get_numerical_xaxis(x)
        assert len(cx.dims) == 1
        assert cx.dims[0] == x.dims[0]
        assert cx.attrs["units"] == x.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cx.values[i], x.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cx.values[nb] == x.values[0]
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], x.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == x.values[0]

        cx = hb.get_numerical_xaxis(xu)
        assert len(cx.dims) == 1
        assert cx.dims[0] == xu.dims[0]
        assert cx.attrs["units"] == xu.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 2
        assert all([eq(cx.values[i], xu.values[0] - nb + i) for i in range(nb)])
        assert cx.values[nb] == xu.values[0]
        assert cx.values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == xu.values[0]
        assert cx.coords[cx.dims[0]].values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(x, hb.get_numerical_xaxis(x))
        compare_dataarrays(xu, hb.get_numerical_xaxis(xu))

    compare_dataarrays(x, hb.get_physical_xaxis(hb.get_numerical_xaxis(x)))
    compare_dataarrays(xu, hb.get_physical_xaxis(hb.get_numerical_xaxis(xu)))

    #
    # y-axis
    #
    y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

    if ny == 1:
        cy = hb.get_numerical_yaxis(y)
        assert len(cy.dims) == 1
        assert cy.dims[0] == y.dims[0]
        assert cy.attrs["units"] == y.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cy.values[i], y.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cy.values[nb] == y.values[0]
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], y.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == y.values[0]

        cy = hb.get_numerical_yaxis(yv)
        assert len(cy.dims) == 1
        assert cy.dims[0] == yv.dims[0]
        assert cy.attrs["units"] == yv.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 2
        assert all([eq(cy.values[i], yv.values[0] - nb + i) for i in range(nb)])
        assert cy.values[nb] == yv.values[0]
        assert cy.values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == yv.values[0]
        assert cy.coords[cy.dims[0]].values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(y, hb.get_numerical_yaxis(y))
        compare_dataarrays(yv, hb.get_numerical_yaxis(yv))

    compare_dataarrays(y, hb.get_physical_yaxis(hb.get_numerical_yaxis(y)))
    compare_dataarrays(yv, hb.get_physical_yaxis(hb.get_numerical_yaxis(yv)))

    #
    # numerical and physical field
    #
    field = state["x_velocity"].values

    if ny == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=1)
        )
    elif nx == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=0)
        )
        compare_arrays(hb.get_numerical_field(field), field)

    compare_arrays(hb.get_physical_field(hb.get_numerical_field(field)), field)

    #
    # reference state
    #
    hb.reference_state = cstate

    for key in cstate:
        with subtests.test(key=key):
            if key != "time":
                compare_dataarrays(
                    cstate[key], hb.reference_state[key], compare_coordinate_values=False
                )

    #
    # enforce_field
    #
    field = cstate["x_velocity"].values
    field_dc = deepcopy(field)
    units = cstate["x_velocity"].attrs["units"]
    hb.enforce_field(field, "x_velocity", units, cstate["time"], cgrid)

    compare_arrays(field[nb:-nb, nb:-nb], field_dc[nb:-nb, nb:-nb])

    #
    # enforce_raw
    #
    rcstate = {"time": cstate["time"]}
    for key in state:
        if key != "time":
            rcstate[key] = cstate[key].values

    hb.enforce_raw(rcstate, grid=cgrid)

    #
    # enforce
    #
    hb.enforce(cstate, grid=cgrid)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def test_identity(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(), label="grid")
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("identity", nx, ny, nb), label="hb_kwargs"
    )
    hb = HorizontalBoundary.factory("identity", nx, ny, nb, **hb_kwargs)

    state0 = data.draw(
        st_isentropic_state(grid, moist=True, precipitation=True), label="state0"
    )
    state1 = data.draw(
        st_isentropic_state(grid, moist=True, precipitation=True), label="state1"
    )

    # ========================================
    # test
    # ========================================
    if ny == 1:
        assert isinstance(hb, Identity1DX)
    elif nx == 1:
        assert isinstance(hb, Identity1DY)
    else:
        assert isinstance(hb, Identity)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb

    if ny == 1:
        assert hb.ni == nx
        assert hb.nj == 2 * nb + 1
    elif nx == 1:
        assert hb.ni == 2 * nb + 1
        assert hb.nj == ny
    else:
        assert hb.ni == nx
        assert hb.nj == ny

    assert hb.type == "identity"

    assert len(hb.kwargs) == 0

    #
    # x-axis
    #
    x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

    if nx == 1:
        cx = hb.get_numerical_xaxis(x)
        assert len(cx.dims) == 1
        assert cx.dims[0] == x.dims[0]
        assert cx.attrs["units"] == x.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cx.values[i], x.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cx.values[nb] == x.values[0]
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], x.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == x.values[0]

        cx = hb.get_numerical_xaxis(xu)
        assert len(cx.dims) == 1
        assert cx.dims[0] == xu.dims[0]
        assert cx.attrs["units"] == xu.attrs["units"]
        assert cx.values.shape[0] == 2 * nb + 2
        assert all([eq(cx.values[i], xu.values[0] - nb + i) for i in range(nb)])
        assert cx.values[nb] == xu.values[0]
        assert cx.values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cx.coords) == 1
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cx.coords[cx.dims[0]].values[nb] == xu.values[0]
        assert cx.coords[cx.dims[0]].values[nb + 1] == xu.values[1]
        assert all(
            [
                eq(cx.coords[cx.dims[0]].values[i], xu.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(x, hb.get_numerical_xaxis(x))
        compare_dataarrays(xu, hb.get_numerical_xaxis(xu))

    compare_dataarrays(x, hb.get_physical_xaxis(hb.get_numerical_xaxis(x)))
    compare_dataarrays(xu, hb.get_physical_xaxis(hb.get_numerical_xaxis(xu)))

    #
    # y-axis
    #
    y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

    if ny == 1:
        cy = hb.get_numerical_yaxis(y)
        assert len(cy.dims) == 1
        assert cy.dims[0] == y.dims[0]
        assert cy.attrs["units"] == y.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 1
        assert all(
            [eq(cy.values[i], y.values[0] - nb + i) for i in range(2 * nb + 1) if i != nb]
        )
        assert cy.values[nb] == y.values[0]
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], y.values[0] - nb + i)
                for i in range(2 * nb + 1)
                if i != nb
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == y.values[0]

        cy = hb.get_numerical_yaxis(yv)
        assert len(cy.dims) == 1
        assert cy.dims[0] == yv.dims[0]
        assert cy.attrs["units"] == yv.attrs["units"]
        assert cy.values.shape[0] == 2 * nb + 2
        assert all([eq(cy.values[i], yv.values[0] - nb + i) for i in range(nb)])
        assert cy.values[nb] == yv.values[0]
        assert cy.values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
        assert len(cy.coords) == 1
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[0] - nb + i)
                for i in range(nb)
            ]
        )
        assert cy.coords[cy.dims[0]].values[nb] == yv.values[0]
        assert cy.coords[cy.dims[0]].values[nb + 1] == yv.values[1]
        assert all(
            [
                eq(cy.coords[cy.dims[0]].values[i], yv.values[1] - nb - 1 + i)
                for i in range(nb + 2, 2 * nb + 2)
            ]
        )
    else:
        compare_dataarrays(y, hb.get_numerical_yaxis(y))
        compare_dataarrays(yv, hb.get_numerical_yaxis(yv))

    compare_dataarrays(y, hb.get_physical_yaxis(hb.get_numerical_yaxis(y)))
    compare_dataarrays(yv, hb.get_physical_yaxis(hb.get_numerical_yaxis(yv)))

    #
    # numerical and physical field
    #
    field = state0["air_isentropic_density"].values
    field_stgx = state0["x_velocity_at_u_locations"].values
    field_stgy = state0["y_velocity_at_v_locations"].values

    if ny == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=1)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx), np.repeat(field_stgx, 2 * nb + 1, axis=1)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy)[:, : nb + 1, :],
            np.repeat(field_stgy[:, 0:1, :], nb + 1, axis=1),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy)[:, nb + 1 :, :],
            np.repeat(field_stgy[:, -1:, :], nb + 1, axis=1),
        )
    elif nx == 1:
        compare_arrays(
            hb.get_numerical_field(field), np.repeat(field, 2 * nb + 1, axis=0)
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx)[: nb + 1, :, :],
            np.repeat(field_stgx[0:1, :, :], nb + 1, axis=0),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgx)[nb + 1 :, :, :],
            np.repeat(field_stgx[-1:, :, :], nb + 1, axis=0),
        )
        compare_arrays(
            hb.get_numerical_field(field_stgy), np.repeat(field_stgy, 2 * nb + 1, axis=0)
        )
    else:
        compare_arrays(hb.get_numerical_field(field), field)
        compare_arrays(hb.get_numerical_field(field_stgx), field_stgx)
        compare_arrays(hb.get_numerical_field(field_stgy), field_stgy)

    compare_arrays(hb.get_physical_field(hb.get_numerical_field(field)), field)

    #
    # reference state
    #
    fake_domain = type("FakeDomain", (object,), {})()
    fake_domain.numerical_grid = NumericalGrid(grid, hb)
    fake_domain.horizontal_boundary = hb

    cstate0 = get_numerical_state(fake_domain, state0)

    hb.reference_state = cstate0

    for key in cstate0:
        with subtests.test(key=key):
            if key != "time":
                compare_dataarrays(
                    cstate0[key], hb.reference_state[key], compare_coordinate_values=False
                )

    #
    # enforce_field
    #
    field_names = (
        "air_isentropic_density",
        "x_velocity_at_u_locations",
        "y_velocity_at_v_locations",
        "air_pressure_on_interface_levels",
        "precipitation",
    )

    cstate1 = get_numerical_state(fake_domain, state1, store_names=field_names)

    for name in field_names:
        with subtests.test(name=name):
            field = cstate1[name].values
            field_dc = deepcopy(field)
            units = cstate1[name].attrs["units"]

            hb.enforce_field(field, name, units, cstate1["time"])

            if ny == 1:
                compare_arrays(field[:, nb:-nb], field_dc[:, nb:-nb])
                for i in range(nb):
                    compare_arrays(field[:, i], field[:, nb])
                    compare_arrays(field[:, -i - 1], field[:, -nb - 1])
            elif nx == 1:
                compare_arrays(field[nb:-nb, :], field_dc[nb:-nb, :])
                for i in range(nb):
                    compare_arrays(field[i, :], field[nb, :])
                    compare_arrays(field[-i - 1, :], field[-nb - 1, :])
            else:
                compare_arrays(field, field_dc)


if __name__ == "__main__":
    pytest.main([__file__])
