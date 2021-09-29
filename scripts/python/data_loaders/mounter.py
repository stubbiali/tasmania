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
from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.io import load_netcdf_dataset


class DatasetMounter:
    ledger = {}
    order = {}

    def __new__(cls, filename: str) -> "DatasetMounter":
        if filename not in DatasetMounter.ledger:
            DatasetMounter.ledger[filename] = super().__new__(cls)
            DatasetMounter.order[filename] = len(DatasetMounter.ledger)
            print(
                f"Instance #{DatasetMounter.order[filename]:03d} of "
                f"DatasetMounter: created."
            )
        return DatasetMounter.ledger[filename]

    def __init__(self, filename: str) -> None:
        if not hasattr(self, "mounted"):
            self.fname = filename
            domain, grid_type, self.states = load_netcdf_dataset(filename)
            print(
                f"Instance #{DatasetMounter.order[filename]:03d} of "
                f"DatasetMounter: dataset mounted."
            )
            self.grid = (
                domain.physical_grid
                if grid_type == "physical"
                else domain.numerical_grid
            )
            self.mounted = True  # tag
        else:
            print(
                f"Instance #{DatasetMounter.order[filename]:03d} of "
                f"DatasetMounter: dataset already mounted."
            )

    def get_grid(self) -> Grid:
        return self.grid

    def get_nt(self) -> int:
        return len(self.states)

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        state = self.states[tlevel]
        self.grid.update_topography(state["time"] - self.states[0]["time"])
        return state
