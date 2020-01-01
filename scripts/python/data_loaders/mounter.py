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
import tasmania as taz


class DatasetMounter:
    ledger = {}

    def __new__(cls, filename):
        if filename not in DatasetMounter.ledger:
            DatasetMounter.ledger[filename] = super().__new__(cls)
            print("New instance of DatasetMounter created.")
        return DatasetMounter.ledger[filename]

    def __init__(self, filename):
        if not hasattr(self, "mounted"):
            self.fname = filename
            domain, grid_type, self.states = taz.load_netcdf_dataset(filename)
            print("  Dataset mounted.")
            self.grid = (
                domain.physical_grid if grid_type == "physical" else domain.numerical_grid
            )
            self.mounted = True  # tag

    def get_grid(self):
        return self.grid

    def get_nt(self):
        return len(self.states)

    def get_state(self, tlevel):
        state = self.states[tlevel]
        self.grid.update_topography(state["time"] - self.states[0]["time"])
        return state
