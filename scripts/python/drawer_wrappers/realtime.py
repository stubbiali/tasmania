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

from tasmania.python.plot.contour import Contour
from tasmania.python.plot.contourf import Contourf
from tasmania.python.plot.profile import LineProfile
from tasmania.python.plot.quiver import Quiver

from scripts.python.data_loaders.base import BaseLoader
from scripts.python.drawer_wrappers.base import DrawerWrapper


class ContourWrapper(DrawerWrapper):
    def __init__(self, loader: BaseLoader, json_filename: str) -> None:
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            field_name = data["field_name"]
            field_units = data["field_units"]

            x = data.get("x", None)
            y = data.get("y", None)
            z = data.get("z", None)

            xaxis_name = data.get("xaxis_name", None)
            xaxis_units = data.get("xaxis_units", None)
            xaxis_y = data.get("xaxis_y", None)
            xaxis_z = data.get("xaxis_z", None)

            yaxis_name = data.get("yaxis_name", None)
            yaxis_units = data.get("yaxis_units", None)
            yaxis_x = data.get("yaxis_x", None)
            yaxis_z = data.get("yaxis_z", None)

            zaxis_name = data.get("zaxis_name", None)
            zaxis_units = data.get("zaxis_units", None)
            zaxis_x = data.get("zaxis_x", None)
            zaxis_y = data.get("zaxis_y", None)

            drawer_properties = data["drawer_properties"]

            self.core = Contour(
                loader.get_grid(),
                field_name,
                field_units,
                x=x,
                y=y,
                z=z,
                xaxis_name=xaxis_name,
                xaxis_units=xaxis_units,
                xaxis_y=xaxis_y,
                xaxis_z=xaxis_z,
                yaxis_name=yaxis_name,
                yaxis_units=yaxis_units,
                yaxis_x=yaxis_x,
                yaxis_z=yaxis_z,
                zaxis_name=zaxis_name,
                zaxis_units=zaxis_units,
                zaxis_x=zaxis_x,
                zaxis_y=zaxis_y,
                properties=drawer_properties,
            )


class ContourfWrapper(DrawerWrapper):
    def __init__(self, loader: BaseLoader, json_filename: str) -> None:
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            field_name = data["field_name"]
            field_units = data["field_units"]

            x = data.get("x", None)
            y = data.get("y", None)
            z = data.get("z", None)

            xaxis_name = data.get("xaxis_name", None)
            xaxis_units = data.get("xaxis_units", None)
            xaxis_y = data.get("xaxis_y", None)
            xaxis_z = data.get("xaxis_z", None)

            yaxis_name = data.get("yaxis_name", None)
            yaxis_units = data.get("yaxis_units", None)
            yaxis_x = data.get("yaxis_x", None)
            yaxis_z = data.get("yaxis_z", None)

            zaxis_name = data.get("zaxis_name", None)
            zaxis_units = data.get("zaxis_units", None)
            zaxis_x = data.get("zaxis_x", None)
            zaxis_y = data.get("zaxis_y", None)

            drawer_properties = data["drawer_properties"]

            self.core = Contourf(
                loader.get_grid(),
                field_name,
                field_units,
                x=x,
                y=y,
                z=z,
                xaxis_name=xaxis_name,
                xaxis_units=xaxis_units,
                xaxis_y=xaxis_y,
                xaxis_z=xaxis_z,
                yaxis_name=yaxis_name,
                yaxis_units=yaxis_units,
                yaxis_x=yaxis_x,
                yaxis_z=yaxis_z,
                zaxis_name=zaxis_name,
                zaxis_units=zaxis_units,
                zaxis_x=zaxis_x,
                zaxis_y=zaxis_y,
                properties=drawer_properties,
            )


class LineProfileWrapper(DrawerWrapper):
    def __init__(self, loader: BaseLoader, json_filename: str) -> None:
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            field_name = data["field_name"]
            field_units = data["field_units"]

            x = data.get("x", None)
            y = data.get("y", None)
            z = data.get("z", None)

            axis_name = data.get("axis_name", None)
            axis_units = data.get("axis_units", None)
            axis_x = data.get("axis_x", None)
            axis_y = data.get("axis_y", None)
            axis_z = data.get("axis_z", None)

            drawer_properties = data["drawer_properties"]

            self.core = LineProfile(
                loader.get_grid(),
                field_name,
                field_units,
                x=x,
                y=y,
                z=z,
                axis_name=axis_name,
                axis_units=axis_units,
                axis_x=axis_x,
                axis_y=axis_y,
                axis_z=axis_z,
                properties=drawer_properties,
            )


class QuiverWrapper(DrawerWrapper):
    def __init__(self, loader: BaseLoader, json_filename: str) -> None:
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)

            x = data.get("x", None)
            y = data.get("y", None)
            z = data.get("z", None)

            xcomp_name = data.get("xcomp_name", None)
            xcomp_units = data.get("xcomp_units", None)

            ycomp_name = data.get("ycomp_name", None)
            ycomp_units = data.get("ycomp_units", None)

            zcomp_name = data.get("zcomp_name", None)
            zcomp_units = data.get("zcomp_units", None)

            scalar_name = data.get("scalar_name", None)
            scalar_units = data.get("scalar_units", None)

            xaxis_name = data.get("xaxis_name", None)
            xaxis_units = data.get("xaxis_units", None)
            xaxis_y = data.get("xaxis_y", None)
            xaxis_z = data.get("xaxis_z", None)

            yaxis_name = data.get("yaxis_name", None)
            yaxis_units = data.get("yaxis_units", None)
            yaxis_x = data.get("yaxis_x", None)
            yaxis_z = data.get("yaxis_z", None)

            zaxis_name = data.get("zaxis_name", None)
            zaxis_units = data.get("zaxis_units", None)
            zaxis_x = data.get("zaxis_x", None)
            zaxis_y = data.get("zaxis_y", None)

            drawer_properties = data["drawer_properties"]

            self.core = Quiver(
                loader.get_grid(),
                x=x,
                y=y,
                z=z,
                xcomp_name=xcomp_name,
                xcomp_units=xcomp_units,
                ycomp_name=ycomp_name,
                ycomp_units=ycomp_units,
                zcomp_units=zcomp_units,
                zcomp_name=zcomp_name,
                scalar_name=scalar_name,
                scalar_units=scalar_units,
                xaxis_name=xaxis_name,
                xaxis_units=xaxis_units,
                xaxis_y=xaxis_y,
                xaxis_z=xaxis_z,
                yaxis_name=yaxis_name,
                yaxis_units=yaxis_units,
                yaxis_x=yaxis_x,
                yaxis_z=yaxis_z,
                zaxis_name=zaxis_name,
                zaxis_units=zaxis_units,
                zaxis_x=zaxis_x,
                zaxis_y=zaxis_y,
                properties=drawer_properties,
            )
