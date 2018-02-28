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
""" 
Test Topography2d class. 
"""
import numpy as np

from grids.grid_xy import GridXY
from grids.topography import Topography2d

domain_x, nx = [0.,10.], 1e2+1
domain_y, ny = [0.,10.], 1e2+1

grid = GridXY(domain_x, nx, domain_y, ny)

topo_str = '3000. * exp(- (x-3.)*(x-3.) - (y-5.)(y-5.))'

hs = Topography2d(grid, 
				  #topo_type = 'schaer', topo_width_x = 1., topo_width_y = 2.)
				  topo_type = 'user_defined', topo_str = topo_str)

hs.plot(grid)

print('Test passed!')
