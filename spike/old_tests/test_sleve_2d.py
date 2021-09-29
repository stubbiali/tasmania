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
Test SLEVE2d class. Data are retrieved from COSMO documentation, Part I.
"""

from python.grids import SLEVE2d as Grid

# Define zonal and vertical domain
domain_x, nx = [0.0, 100.0], 101
domain_z, nz = [15000.0, 0.0], 20

# Specify the interfacial height separating the terrain-following part of
# the domain from the z-system
zf = 11360.0

# Define terrain-surface height
topo_str = "3000. * exp(- (x - 50.)*(x - 50.) / (20.*20.) )"

# Instantiate a grid object
g = Grid(
    domain_x,
    nx,
    domain_z,
    nz,
    interface_z=zf,
    # topo_type = 'gaussian', topo_max_height = 3000., topo_width_x = 20)
    topo_type="user_defined",
    topo_str=topo_str,
)

# Plot
g.plot()

print("Test passed!")
