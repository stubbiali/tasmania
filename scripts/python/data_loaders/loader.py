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

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty

from scripts.python.data_loaders.base import BaseLoader
from scripts.python.data_loaders.mounter import DatasetMounter


class Loader(BaseLoader):
    def __init__(self, json_filename: str) -> None:
        with open(json_filename, "r") as json_file:
            data = json.load(json_file)
            filename = "".join(data["filename"])
            self.dsmounter = DatasetMounter(filename)

    def get_grid(self) -> Grid:
        return self.dsmounter.get_grid()

    def get_nt(self) -> int:
        return self.dsmounter.get_nt()

    def get_initial_time(self) -> ty.Datetime:
        return self.dsmounter.get_state(0)["time"]

    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        return self.dsmounter.get_state(tlevel)
