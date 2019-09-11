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
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.burgers.dynamics.advection import (
    BurgersAdvection,
    _FirstOrder,
    _SecondOrder,
    _ThirdOrder,
    _FourthOrder,
    _FifthOrder,
)

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import compare_arrays, st_burgers_state, st_one_of, st_physical_grid
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import compare_arrays, st_burgers_state, st_one_of, st_physical_grid


class WrappingStencil:
    def __init__(self, advection, nb, backend):
        assert nb >= advection.extent
        self.nb = nb
        decorator = gt.stencil(
            backend, rebuild=True, externals={"call_func": advection.__call__}
        )
        self.stencil = decorator(self.stencil_defs)

    def __call__(self, dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y):
        mi, mj, mk = u.data.shape
        nb = self.nb
        self.stencil(
            in_u=u,
            in_v=v,
            out_adv_u_x=adv_u_x,
            out_adv_u_y=adv_u_y,
            out_adv_v_x=adv_v_x,
            out_adv_v_y=adv_v_y,
            dx=dx,
            dy=dy,
            origin={"_all_": (nb, nb, 0)},
            domain=(mi - 2 * nb, mj - 2 * nb, mk),
        )

    @staticmethod
    def stencil_defs(
        in_u: gt.storage.f64_ijk_sd,
        in_v: gt.storage.f64_ijk_sd,
        out_adv_u_x: gt.storage.f64_ijk_sd,
        out_adv_u_y: gt.storage.f64_ijk_sd,
        out_adv_v_x: gt.storage.f64_ijk_sd,
        out_adv_v_y: gt.storage.f64_ijk_sd,
        *,
        dx: float,
        dy: float
    ):
        out_adv_u_x, out_adv_u_y, out_adv_v_x, out_adv_v_y = call_func(
            dx=dx, dy=dy, u=in_u, v=in_v
        )


def first_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi, dtype=phi.dtype)
    adv_x[1:-1, :, :] = u[1:-1, :, :] / (2.0 * dx) * (
        phi[2:, :, :] - phi[:-2, :, :]
    ) - np.abs(u)[1:-1, :, :] / (2.0 * dx) * (
        phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]
    )
    adv_y = np.zeros_like(phi, dtype=phi.dtype)
    adv_y[:, 1:-1, :] = v[:, 1:-1, :] / (2.0 * dy) * (
        phi[:, 2:, :] - phi[:, :-2, :]
    ) - np.abs(v)[:, 1:-1, :] / (2.0 * dy) * (
        phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
    )
    return adv_x, adv_y


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_first_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
        ),
        label="grid",
    )
    state = data.draw(st_burgers_state(grid), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test test
    # ========================================
    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").values
    v = state["y_velocity"].to_units("m s^-1").values

    dtype = u.dtype
    grid_shape = u.shape
    halo = tuple(halo[i] if grid_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(grid_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    adv_u_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_u_y_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_y_st = gt.storage.empty(descriptor, backend=backend)

    advection = BurgersAdvection.factory("first_order")

    ws = WrappingStencil(advection, nb, backend)

    ws(dx, dy, u_st, v_st, adv_u_x_st, adv_u_y_st, adv_v_x_st, adv_v_y_st)

    adv_u_x, adv_u_y = first_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u, v, v)

    compare_arrays(adv_u_x_st.data[nb:-nb, nb:-nb], adv_u_x[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y_st.data[nb:-nb, nb:-nb], adv_u_y[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x_st.data[nb:-nb, nb:-nb], adv_v_x[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y_st.data[nb:-nb, nb:-nb], adv_v_y[nb:-nb, nb:-nb])


def second_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi, dtype=phi.dtype)
    adv_x[1:-1, :, :] = u[1:-1, :, :] / (2.0 * dx) * (phi[2:, :, :] - phi[:-2, :, :])
    adv_y = np.zeros_like(phi, dtype=phi.dtype)
    adv_y[:, 1:-1, :] = v[:, 1:-1, :] / (2.0 * dy) * (phi[:, 2:, :] - phi[:, :-2, :])
    return adv_x, adv_y


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_second_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
        ),
        label="grid",
    )
    state = data.draw(st_burgers_state(grid), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test test
    # ========================================
    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").values
    v = state["y_velocity"].to_units("m s^-1").values

    dtype = u.dtype
    grid_shape = u.shape
    halo = tuple(halo[i] if grid_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(grid_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    adv_u_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_u_y_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_y_st = gt.storage.empty(descriptor, backend=backend)

    advection = BurgersAdvection.factory("second_order")

    ws = WrappingStencil(advection, nb, backend)

    ws(dx, dy, u_st, v_st, adv_u_x_st, adv_u_y_st, adv_v_x_st, adv_v_y_st)

    adv_u_x, adv_u_y = second_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = second_order_advection(dx, dy, u, v, v)

    compare_arrays(adv_u_x_st.data[nb:-nb, nb:-nb], adv_u_x[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y_st.data[nb:-nb, nb:-nb], adv_u_y[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x_st.data[nb:-nb, nb:-nb], adv_v_x[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y_st.data[nb:-nb, nb:-nb], adv_v_y[nb:-nb, nb:-nb])


def third_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi, dtype=phi.dtype)
    adv_x[2:-2, :, :] = u[2:-2, :, :] / (12.0 * dx) * (
        8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :]) - (phi[4:, :, :] - phi[:-4, :, :])
    ) + np.abs(u)[2:-2, :, :] / (12.0 * dx) * (
        (phi[4:, :, :] + phi[:-4, :, :])
        - 4.0 * (phi[3:-1, :, :] + phi[1:-3, :, :])
        + 6.0 * phi[2:-2, :, :]
    )
    adv_y = np.zeros_like(phi, dtype=phi.dtype)
    adv_y[:, 2:-2, :] = v[:, 2:-2, :] / (12.0 * dy) * (
        8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :]) - (phi[:, 4:, :] - phi[:, :-4, :])
    ) + np.abs(v)[:, 2:-2, :] / (12.0 * dy) * (
        (phi[:, 4:, :] + phi[:, :-4, :])
        - 4.0 * (phi[:, 3:-1, :] + phi[:, 1:-3, :])
        + 6.0 * phi[:, 2:-2, :]
    )
    return adv_x, adv_y


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_third_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
        ),
        label="grid",
    )
    state = data.draw(st_burgers_state(grid), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test test
    # ========================================
    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").values
    v = state["y_velocity"].to_units("m s^-1").values

    dtype = u.dtype
    grid_shape = u.shape
    halo = tuple(halo[i] if grid_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(grid_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    adv_u_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_u_y_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_y_st = gt.storage.empty(descriptor, backend=backend)

    advection = BurgersAdvection.factory("third_order")

    ws = WrappingStencil(advection, nb, backend)

    ws(dx, dy, u_st, v_st, adv_u_x_st, adv_u_y_st, adv_v_x_st, adv_v_y_st)

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u, v, v)

    compare_arrays(adv_u_x_st.data[nb:-nb, nb:-nb], adv_u_x[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y_st.data[nb:-nb, nb:-nb], adv_u_y[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x_st.data[nb:-nb, nb:-nb], adv_v_x[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y_st.data[nb:-nb, nb:-nb], adv_v_y[nb:-nb, nb:-nb])


def fourth_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi, dtype=phi.dtype)
    adv_x[2:-2, :, :] = (
        u[2:-2, :, :]
        / (12.0 * dx)
        * (8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :]) - (phi[4:, :, :] - phi[:-4, :, :]))
    )
    adv_y = np.zeros_like(phi, dtype=phi.dtype)
    adv_y[:, 2:-2, :] = (
        v[:, 2:-2, :]
        / (12.0 * dy)
        * (8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :]) - (phi[:, 4:, :] - phi[:, :-4, :]))
    )
    return adv_x, adv_y


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_fourth_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
        ),
        label="grid",
    )
    state = data.draw(st_burgers_state(grid), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test test
    # ========================================
    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").values
    v = state["y_velocity"].to_units("m s^-1").values

    dtype = u.dtype
    grid_shape = u.shape
    halo = tuple(halo[i] if grid_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(grid_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    adv_u_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_u_y_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_y_st = gt.storage.empty(descriptor, backend=backend)

    advection = BurgersAdvection.factory("fourth_order")

    ws = WrappingStencil(advection, nb, backend)

    ws(dx, dy, u_st, v_st, adv_u_x_st, adv_u_y_st, adv_v_x_st, adv_v_y_st)

    adv_u_x, adv_u_y = fourth_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = fourth_order_advection(dx, dy, u, v, v)

    compare_arrays(adv_u_x_st.data[nb:-nb, nb:-nb], adv_u_x[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y_st.data[nb:-nb, nb:-nb], adv_u_y[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x_st.data[nb:-nb, nb:-nb], adv_v_x[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y_st.data[nb:-nb, nb:-nb], adv_v_y[nb:-nb, nb:-nb])


def fifth_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi, dtype=phi.dtype)
    adv_x[3:-3, :, :] = u[3:-3, :, :] / (60.0 * dx) * (
        45.0 * (phi[4:-2, :, :] - phi[2:-4, :, :])
        - 9.0 * (phi[5:-1, :, :] - phi[1:-5, :, :])
        + (phi[6:, :, :] - phi[:-6, :, :])
    ) - np.abs(u)[3:-3, :, :] / (60.0 * dx) * (
        (phi[6:, :, :] + phi[:-6, :, :])
        - 6.0 * (phi[5:-1, :, :] + phi[1:-5, :, :])
        + 15.0 * (phi[4:-2, :, :] + phi[2:-4, :, :])
        - 20.0 * phi[3:-3, :, :]
    )
    adv_y = np.zeros_like(phi, dtype=phi.dtype)
    adv_y[:, 3:-3, :] = v[:, 3:-3, :] / (60.0 * dy) * (
        45.0 * (phi[:, 4:-2, :] - phi[:, 2:-4, :])
        - 9.0 * (phi[:, 5:-1, :] - phi[:, 1:-5, :])
        + (phi[:, 6:, :] - phi[:, :-6, :])
    ) - np.abs(v)[:, 3:-3, :] / (60.0 * dy) * (
        (phi[:, 6:, :] + phi[:, :-6, :])
        - 6.0 * (phi[:, 5:-1, :] + phi[:, 1:-5, :])
        + 15.0 * (phi[:, 4:-2, :] + phi[:, 2:-4, :])
        - 20.0 * phi[:, 3:-3, :]
    )
    return adv_x, adv_y


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_fifth_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb")
    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
        ),
        label="grid",
    )
    state = data.draw(st_burgers_state(grid), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test test
    # ========================================
    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").values
    v = state["y_velocity"].to_units("m s^-1").values

    dtype = u.dtype
    grid_shape = u.shape
    halo = tuple(halo[i] if grid_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(grid_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(dtype, halo=halo, iteration_domain=domain)
    u_st = gt.storage.from_array(u, descriptor, backend=backend)
    v_st = gt.storage.from_array(v, descriptor, backend=backend)
    adv_u_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_u_y_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_x_st = gt.storage.empty(descriptor, backend=backend)
    adv_v_y_st = gt.storage.empty(descriptor, backend=backend)

    advection = BurgersAdvection.factory("fifth_order")

    ws = WrappingStencil(advection, nb, backend)

    ws(dx, dy, u_st, v_st, adv_u_x_st, adv_u_y_st, adv_v_x_st, adv_v_y_st)

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u, v, v)

    compare_arrays(adv_u_x_st.data[nb:-nb, nb:-nb], adv_u_x[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y_st.data[nb:-nb, nb:-nb], adv_u_y[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x_st.data[nb:-nb, nb:-nb], adv_v_x[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y_st.data[nb:-nb, nb:-nb], adv_v_y[nb:-nb, nb:-nb])


if __name__ == "__main__":
    # pytest.main([__file__])
    test_first_order()
