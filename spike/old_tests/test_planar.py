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
## @package gt4ess
#  Test Planar class.

from python.grids import Planar as Grid

#
# Data
#
kwargs = dict(
    domain_x=[0.0, 10.0],
    nx=101,
    units_x="km",
    dims_x="x",
    domain_y=[-5.0, 7.0],
    ny=241,
    units_y="km",
    dims_y="y",
)

#
# Test
#
g = Grid(**kwargs)
