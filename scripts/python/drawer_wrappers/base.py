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
import abc
from tasmania import taz_types
from tasmania.python.plot.drawer import Drawer

from scripts.python.data_loaders.base import BaseLoader


class DrawerWrapper(abc.ABC):
    def __init__(self, loader: BaseLoader) -> None:
        self.loader = loader
        self.core = None

    def get_drawer(self) -> Drawer:
        return self.core

    def get_initial_time(self) -> taz_types.datetime_t:
        return self.loader.get_initial_time()

    def get_state(self, tlevel: int) -> taz_types.dataarray_dict_t:
        return self.loader.get_state(tlevel)
