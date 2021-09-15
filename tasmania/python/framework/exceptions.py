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
from sympl._core.units import clean_units


class CouplingError(Exception):
    pass


class IncompatibleDimensionsError(Exception):
    def __init__(self, dim1, dim2):
        super().__init__(
            f"Dimensions ({', '.join(dim1)}) and ({', '.join(dim2)}) "
            f"are not compatible."
        )


class IncompatibleUnitsError(Exception):
    def __init__(self, unit1, unit2):
        super().__init__(
            f"Units [{clean_units(unit1)}] and [{clean_units(unit2)}] "
            f"are not compatible."
        )
