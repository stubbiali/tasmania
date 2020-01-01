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
import tasmania as taz

try:
    from .base import DrawerWrapper
except (ImportError, ModuleNotFoundError):
    from base import DrawerWrapper


class AnnotationWrapper(DrawerWrapper):
    def __init__(self, loader, json_filename):
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            drawer_properties = data["drawer_properties"]
            self.core = taz.Annotation(drawer_properties)


class CircleWrapper(DrawerWrapper):
    def __init__(self, loader, json_filename):
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            drawer_properties = data["drawer_properties"]
            self.core = taz.Circle(drawer_properties)


class RectangleWrapper(DrawerWrapper):
    def __init__(self, loader, json_filename):
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            drawer_properties = data["drawer_properties"]
            self.core = taz.Rectangle(drawer_properties)


class SegmentWrapper(DrawerWrapper):
    def __init__(self, loader, json_filename):
        super().__init__(loader)

        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            x = data["x"]
            y = data["y"]
            drawer_properties = data["drawer_properties"]
            self.core = taz.Segment(x, y, drawer_properties)
