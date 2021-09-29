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

from tasmania.python.plot.offline import Line
from tasmania.python.utils import typingx as ty

from scripts.python.data_loaders.composite import CompositeLoader
from scripts.python.drawer_wrappers.base import DrawerWrapper


class LineWrapper(DrawerWrapper):
    def __init__(self, loader: CompositeLoader, json_filename: str) -> None:
        # assert isinstance(loader, CompositeLoader)
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            field_name = data["field_name"]
            field_units = data["field_units"]
            x = data["x"]
            y = data["y"]
            z = data["z"]
            xdata = data.get("xdata", None)
            ydata = data.get("ydata", None)
            drawer_properties = data["drawer_properties"]

            self.core = Line(
                loader.get_grid(),
                field_name,
                field_units,
                x,
                y,
                z,
                xdata=xdata,
                ydata=ydata,
                properties=drawer_properties,
            )

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        self.core.reset()
        states = self.loader.get_state(tlevel)
        for state in states[:-1]:
            self.core(state)
        return states[-1]
