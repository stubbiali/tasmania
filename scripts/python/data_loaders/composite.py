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
import json
import numpy as np
from sympl import DataArray
from typing import List

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty

from scripts.python.data_loaders.base import BaseLoader
from scripts.python.data_loaders.diff import RMSDLoader


class CompositeLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            self.slaves = []
            for slave_config in data["slaves"]:
                module = slave_config["module"]
                classname = slave_config["classname"]
                config = slave_config["config"]

                try:
                    exec("from .{} import {}".format(module, classname))
                except (ImportError, ModuleNotFoundError):
                    exec("from {} import {}".format(module, classname))
                SlaveClass = locals()[classname]
                self.slaves.append(SlaveClass(config))

    def get_grid(self) -> List[Grid]:
        grids = []
        for slave in self.slaves:
            grids.append(slave.get_grid())
        return grids

    def get_nt(self) -> List[int]:
        nts = []
        for slave in self.slaves:
            nts.append(slave.get_nt())
        return nts

    def get_initial_time(self) -> List[ty.Datetime]:
        itimes = []
        for slave in self.slaves:
            itimes.append(slave.get_initial_time())
        return itimes

    def get_state(self, tlevel) -> List[ty.DataArrayDict]:
        tlevel = [tlevel] if isinstance(tlevel, int) else tlevel
        tlevels = tlevel * len(self.slaves) if len(tlevel) == 1 else tlevel
        assert len(tlevels) == len(self.slaves)

        states = []
        for slave, tl in zip(self.slaves, tlevels):
            states.append(slave.get_state(tl))

        return states


class PolynomialInterpolationLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            self.slaves = []
            self.x = []
            self.y = []
            self.z = []
            for slave_config in data["slaves"]:
                module = slave_config["loader_module"]
                classname = slave_config["loader_classname"]
                config = slave_config["loader_config"]

                try:
                    exec("from .{} import {}".format(module, classname))
                except (ImportError, ModuleNotFoundError):
                    exec("from {} import {}".format(module, classname))
                SlaveClass = locals()[classname]
                self.slaves.append(SlaveClass(config))

                self.x.append(slave_config["x"])
                self.y.append(slave_config["y"])
                self.z.append(slave_config["z"])

            self.fname = data["field_name"]
            self.funits = data["field_units"]

            self.deg = data["polynomial_degree"]

    def get_grid(self) -> Grid:
        return self.slaves[0].get_grid()

    def get_nt(self) -> int:
        return self.slaves[0].get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.slaves[0].get_nt()

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        slaves = self.slaves
        x, y, z = self.x, self.y, self.z
        fname, funits = self.fname, self.funits
        deg = self.deg

        return_state = slaves[0].get_state(tlevel)

        data_y = []
        for slave, i, j, k in zip(slaves, x, y, z):
            state = slave.get_state(tlevel)
            data_y.append(state[fname].to_units(funits).values[i, j, k].item())
        data_y = np.log2(np.array(data_y))
        data_x = np.arange(len(slaves), 0, -1)

        try:
            fit = np.polyfit(data_x, data_y, deg)
        except (np.linalg.LinAlgError, ValueError):
            fit = [0] * (deg + 1)

        return_state["polyfit_of_" + fname] = DataArray(
            np.array(fit[0])[np.newaxis, np.newaxis, np.newaxis],
            attrs={"units": "1"},
        )

        return return_state


class EOCLoader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            self.loader_c = RMSDLoader(data["config_coarse"])
            self.loader_f = RMSDLoader(data["config_fine"])

    def get_grid(self) -> Grid:
        return self.loader_f.get_grid()

    def get_nt(self) -> int:
        return self.loader_f.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.loader_f.get_initial_time()

    def get_state(self, tlevel) -> ty.DataArrayDict:
        state_c = self.loader_c.get_state(tlevel)
        state_f = self.loader_f.get_state(tlevel)

        fname = self.loader_f.fname
        funits = self.loader_f.funits

        eoc = np.log2(
            state_c["rmsd_of_" + fname].to_units(funits).values[0, 0, 0]
            / state_f["rmsd_of_" + fname].values[0, 0, 0]
        )
        state_f["eoc_of_" + fname] = DataArray(
            np.array(eoc)[np.newaxis, np.newaxis, np.newaxis],
            attrs={"units": "1"},
        )

        return state_f
