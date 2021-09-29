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
import tasmania as taz
from tasmania.python.utils.utils import assert_sequence


class BaseLoader:
    _ledger = {}

    def __new__(cls, filename):
        if filename in BaseLoader._ledger:
            return BaseLoader._ledger[filename]
        else:
            print("A new instance of Loader is going to be created.")
            return super().__new__(cls, filename)

    def __init__(self, filename):
        self._fname = filename
        domain, grid_type, self._states = taz.load_netcdf_dataset(filename)
        self._grid = (
            domain.physical_grid
            if grid_type == "physical"
            else domain.numerical_grid
        )

    def get_nt(self):
        return len(self._states)

    def get_grid(self):
        return self._grid

    def get_state(self, tlevel):
        state = self._states[tlevel]
        self._grid.update_topography(state["time"] - self._states[0]["time"])
        return state


class LoaderComposite:
    def __init__(self, *loaders):
        assert_sequence(loaders, reftype=(BaseLoader, LoaderComposite))
        self._loaders = loaders

    def get_grid(self):
        return_list = []

        for loader in self._loaders:
            return_list.append(loader.get_grid())

        return return_list

    def get_state(self, tlevel):
        return_list = []

        for loader in self._loaders:
            return_list.append(loader.get_state(tlevel))

        return return_list
