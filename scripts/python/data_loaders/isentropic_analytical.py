# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import json
from sympl import DataArray

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.dict import DataArrayDictOperator
from tasmania.python.utils.meteo import (
    get_isothermal_isentropic_analytical_solution,
)

from scripts.python.data_loaders.base import BaseLoader
from scripts.python.data_loaders.mounter import DatasetMounter


class IsentropicAnalyticalLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            filename = "".join(data["filename"])

            self.dsmounter = DatasetMounter(filename)

            self.u = DataArray(
                data["initial_x_velocity"]["value"],
                attrs={"units": data["initial_x_velocity"]["units"]},
            )
            self.t = DataArray(
                data["temperature"]["value"],
                attrs={"units": data["temperature"]["units"]},
            )
            self.h = DataArray(
                data["mountain_height"]["value"],
                attrs={"units": data["mountain_height"]["units"]},
            )
            self.a = DataArray(
                data["mountain_width"]["value"],
                attrs={"units": data["mountain_width"]["units"]},
            )

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        grid = self.get_grid()
        init_state = self.dsmounter.get_state(0)

        u, _ = get_isothermal_isentropic_analytical_solution(
            grid, self.u, self.t, self.h, self.a
        )
        final_state = {"x_velocity_at_u_locations": u}

        op = DataArrayDictOperator(backend="numpy")

        state = op.sub(final_state, init_state)
        state.update(
            {
                key: value
                for key, value in init_state.items()
                if key != "x_velocity_at_u_locations"
            }
        )

        return state
