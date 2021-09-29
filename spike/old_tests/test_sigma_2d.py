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
Test Sigma2d class. Data are retrieved from COSMO documentation, Part I.
"""

import namelist as nl
from python.grids.sigma import Sigma2d as Grid

# Define zonal and vertical domain
domain_x, nx = [0.0, 500.0e3], 51
domain_z, nz = [0.1, 1.0], 50

# Specify the interfacial height separating the terrain-following part of
# the domain from the z-system
pf = 220e2
zf = pf / nl.p_sl

# Define terrain-surface height
# hs = 3000. * np.exp(- (np.linspace(domain_x[0], domain_x[1], n_x) - 5.)**2. / (20.*20.))
hs = "3000. * exp(- (x - 40.)*(x - 40.) / (15.*15.))"

# Instantiate a grid object
g = Grid(
    domain_x,
    nx,
    domain_z,
    nz,  # interface_z = zf,
    topo_type="gaussian",
    topo_max_height=1000.0,
    topo_width_x=50.0e3,
)
# topo_type = 'user_defined', topo_str = hs)

# Plot
g.plot()

print("Test passed!")
