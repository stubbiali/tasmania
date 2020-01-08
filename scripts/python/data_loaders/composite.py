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
import json
import numpy as np
from sympl import DataArray

try:
    from .base import BaseLoader
    from .mounter import DatasetMounter
except ImportError:
    from base import BaseLoader
    from mounter import DatasetMounter


class CompositeLoader(BaseLoader):
    def __init__(self, json_filename):
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

    def get_grid(self):
        grids = []
        for slave in self.slaves:
            grids.append(slave.get_grid())
        return grids

    def get_nt(self):
        nts = []
        for slave in self.slaves:
            nts.append(slave.get_nt())
        return nts

    def get_initial_time(self):
        itimes = []
        for slave in self.slaves:
            itimes.append(slave.get_initial_time())
        return itimes

    def get_state(self, tlevel):
        tlevel = [tlevel,] if isinstance(tlevel, int) else tlevel
        tlevels = tlevel * len(self.slaves) if len(tlevel) == 1 else tlevel
        assert len(tlevels) == len(self.slaves)

        states = []
        for slave, tl in zip(self.slaves, tlevels):
            states.append(slave.get_state(tl))

        return states


class PolynomialInterpolationLoader(BaseLoader):
    def __init__(self, json_filename):
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

    def get_grid(self):
        return self.slaves[0].get_grid()

    def get_nt(self):
        return self.slaves[0].get_nt()

    def get_initial_time(self):
        return self.slaves[0].get_nt()

    def get_state(self, tlevel):
        slaves = self.slaves
        x, y, z = self.x, self.y, self.z
        fname, funits = self.fname, self.funits
        deg = self.deg

        return_state = slaves[0].get_state(tlevel)

        data_y = []
        for slave, i, j, k in zip(slaves, x, y, z):
            state = slave.get_state(tlevel)
            data_y.append(state[fname].to_units(funits).values[i, j, k].item())
        data_x = tuple(data_y[0] / 2 ** i for i in range(len(slaves)))

        try:
            fit = np.polyfit(data_x, data_y, deg)
        except (np.linalg.LinAlgError, ValueError):
            fit = [0] * (deg + 1)

        return_state["polyfit_of_" + fname] = DataArray(
            np.array(fit[0])[np.newaxis, np.newaxis, np.newaxis], attrs={"units": funits}
        )

        return return_state