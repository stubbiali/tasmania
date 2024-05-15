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
import abc

from tasmania.python.domain.grid import Grid
from tasmania.python.utils import typingx as ty


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def get_grid(self) -> Grid:
        pass

    @abc.abstractmethod
    def get_nt(self) -> int:
        pass

    @abc.abstractmethod
    def get_initial_time(self) -> ty.Datetime:
        pass

    @abc.abstractmethod
    def get_state(self, tlevel: int) -> ty.DataArrayDict:
        pass
