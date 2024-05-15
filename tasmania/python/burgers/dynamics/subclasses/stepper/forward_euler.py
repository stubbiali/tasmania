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
from sympl._core.time import Timer

from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.framework.register import register


@register("forward_euler")
class ForwardEuler(BurgersStepper):
    """The forward Euler time integrator."""

    def __init__(
        self,
        grid_xy,
        nb,
        flux_scheme,
        backend,
        backend_options,
        storage_options,
    ):
        super().__init__(
            grid_xy, nb, flux_scheme, backend, backend_options, storage_options
        )

    @property
    def stages(self):
        return 1

    def __call__(self, stage, state, tendencies, timestep, out_state):
        nx, ny = self._grid_xy.nx, self._grid_xy.ny
        nb = self._nb

        if self._forward_euler is None:
            self._stencil_initialize(tendencies)

        dt = timestep.total_seconds()
        dx = self._grid_xy.dx.to_units("m").values.item()
        dy = self._grid_xy.dy.to_units("m").values.item()

        self._stencil_args["in_u"] = state["x_velocity"]
        self._stencil_args["in_u_tmp"] = state["x_velocity"]
        self._stencil_args["in_v"] = state["y_velocity"]
        self._stencil_args["in_v_tmp"] = state["y_velocity"]
        self._stencil_args["out_u"] = out_state["x_velocity"]
        self._stencil_args["out_v"] = out_state["y_velocity"]
        if "x_velocity" in tendencies:
            self._stencil_args["in_u_tnd"] = tendencies["x_velocity"]
        if "y_velocity" in tendencies:
            self._stencil_args["in_v_tnd"] = tendencies["y_velocity"]

        Timer.start(label="stencil")
        self._forward_euler(
            **self._stencil_args,
            dt=dt,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, 1),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args
        )
        Timer.stop()

        out_state["time"] = state["time"] + timestep
