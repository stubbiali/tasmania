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
Test GridXYZ.
"""
from python.grids.grid import GridXYZ

domain_x, nx = [-5., 5,], 101
domain_y, ny = [-5., 5.], 101
domain_z, nz = [300., 300. + 50.], 20

topo_str = '1. / (x*x + 1.)'

xyz_grid = GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz,
				   #topo_type = 'schaer', topo_width_x = 1., topo_width_y = 2.)
				   topo_type = 'user_defined', topo_str = topo_str)

xyz_grid._topography.plot(xyz_grid.xy_grid)

print('Test passed!')
