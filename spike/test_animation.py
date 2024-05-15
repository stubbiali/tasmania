# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
import numpy as np


def _create_grid():
    domain_x, nx = [0, 500e3], 101
    domain_y, ny = [0, 500e3], 91
    domain_z, nz = [400, 300], 50

    from python.grids.grid import GridXYZ as Grid

    grid = Grid(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        units_x="m",
        dims_x="x",
        units_y="m",
        dims_y="y",
        units_z="K",
        dims_z="z",
        topo_type="gaussian",
        topo_max_height=500.0,
        topo_width_x=25.0e3,
        topo_width_y=50.0e3,
        topo_smooth=False,
    )

    return grid


def _create_dataarray(raw_array, grid, units=""):
    # x-axis
    x = grid.x if raw_array.shape[0] == grid.nx else grid.x_at_u_locations

    # y-axis
    y = grid.y if raw_array.shape[1] == grid.ny else grid.y_at_v_locations

    # z-axis
    if len(raw_array.shape) == 2:
        raw_array = raw_array[:, :, np.newaxis]
    if raw_array.shape[2] == 1:
        from tasmana.grids.axis import Axis

        z = Axis(
            np.array([grid.z_on_interface_levels[-1]]),
            grid.z.dims,
            attrs=grid.z.attrs,
        )
    elif raw_array.shape[2] == 1:
        from tasmana.grids.axis import Axis

        z = Axis(
            np.array([grid.z_on_interface_levels[-1]]),
            grid.z.dims,
            attrs=grid.z.attrs,
        )
    elif raw_array.shape[2] == grid.nz:
        z = grid.z
    elif raw_array.shape[2] == grid.nz + 1:
        z = grid.z_on_interface_levels

    # Create the DataArray
    import sympl

    array = sympl.DataArray(
        raw_array,
        coords=[x.values, y.values, z.values],
        dims=[x.dims, y.dims, z.dims],
        attrs={"units": units},
    )

    return array


class _StateCreator:
    def __init__(self):
        self._counter = 0

    def __call__(self, grid):
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        state = {}

        # Initialize time
        from datetime import datetime, timedelta

        state["time"] = datetime(year=1992, month=2, day=20) + timedelta(
            hours=self._counter
        )

        # Initialize density
        np.random.seed(19920220 + 10 * self._counter)
        state["air_density"] = _create_dataarray(
            np.random.rand(nx, ny, nz), grid, units="kg m^-3"
        )

        # Initialize x-velocity
        state["x_velocity_at_u_locations"] = _create_dataarray(
            np.random.rand(nx + 1, ny, nz), grid, units="m s^-1"
        )

        # Compute x-momentum
        rho = state["air_density"].values[:, :, :]
        u = 0.5 * (
            state["x_velocity_at_u_locations"].values[:-1, :, :]
            + state["x_velocity_at_u_locations"].values[1:, :, :]
        )
        state["x_momentum"] = _create_dataarray(
            rho * u, grid, units="kg m^-2 s^-1"
        )

        # Initialize y-velocity
        state["y_velocity_at_v_locations"] = _create_dataarray(
            np.random.rand(nx, ny + 1, nz), grid, units="m s^-1"
        )

        # Compute y-momentum
        v = 0.5 * (
            state["y_velocity_at_v_locations"].values[:, :-1, :]
            + state["y_velocity_at_v_locations"].values[:, 1:, :]
        )
        state["y_momentum"] = _create_dataarray(
            rho * v, grid, units="kg m^-2 s^-1"
        )

        # Initialize pressure
        state["air_preassure_on_interface_levels"] = _create_dataarray(
            np.random.rand(nx, ny, nz + 1), grid, units="Pa"
        )

        # Initialize height
        raw_height = np.zeros((nx, ny, nz + 1))
        raw_height[:, :, -1] = grid.topography_height
        for k in range(nz - 1, -1, -1):
            raw_height[:, :, k] = raw_height[:, :, k + 1] + 100.0
        state["height_on_interface_levels"] = _create_dataarray(
            raw_height, grid, units="m"
        )

        self._counter += 1

        return state


def test_monitor_contour_xz():
    grid = _create_grid()

    state_creator = _StateCreator()
    states_list = []
    for _ in range(30):
        states_list.append(state_creator(grid))

    field_to_plot = "x_velocity_at_u_locations"
    y_level = int(grid.ny / 2)

    kwargs = dict(
        fontsize=16,
        figsize=[7, 8],
        title="test_monitor_contour_xz",
        x_label="$x$",
        x_factor=1e-3,
        x_lim=None,
        z_label="$z$",
        z_factor=1e-3,
        z_lim=[0, 4e3],
        field_bias=0.0,
        field_factor=1.0,
        draw_z_isolines=True,
        fps=10,
        text=field_to_plot,
        text_loc="upper right",
    )

    from python.plot import Animation
    from python.plot import make_animation_contour_xz as plot_function

    monitor = Animation(plot_function, grid, field_to_plot, y_level, **kwargs)
    for state in states_list:
        monitor.store(state)
    monitor.make_animation("/tmp/test_monitor_contour_xz.mp4")


# if __name__ == '__main__':
# 	pytest.main([__file__])
test_monitor_contour_xz()
