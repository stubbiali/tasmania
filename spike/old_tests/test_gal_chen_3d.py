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
"""
Test GalChen3d class.
"""

from python.grids import GalChen3d as Grid

# Define zonal, meridional and vertical domain
domain_x, nx = [0.0, 90.0], 91
domain_y, ny = [-30.0, 45.0], 76
domain_z, nz = [15000.0, 0.0], 20

# Instantiate a grid object
g = Grid(
    domain_x,
    nx,
    domain_y,
    ny,
    domain_z,
    nz,
    topo_type="gaussian",
    topo_width_x=10.0,
    topo_width_y=10.0,
)

print("Test passed!")
