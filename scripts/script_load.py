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
import pickle

filename = os.path.join(os.environ['TASMANIA_ROOT'], 
						'data/datasets/isentropic_convergence_upwind_u10_lx400_nz300_8km_relaxed.pickle')
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	u1 = state_save['x_momentum_isentropic'].values[:, 0, :, -1] / state_save['isentropic_density'].values[:, 0, :, -1]

	grid = state_save.grid
	uex, wex = utils.get_isothermal_solution(grid, 10., 250., 1., 1.e4, x_staggered = False, z_staggered = False)

filename = os.path.join(os.environ['TASMANIA_ROOT'], 
						'data/datasets/isentropic_convergence_upwind_u10_lx400_nz300_4km_relaxed.pickle')
with open(filename, 'rb') as data:
	state_save = pickle.load(data)
	u2 = state_save['x_momentum_isentropic'].values[:, 0, :, -1] / state_save['isentropic_density'].values[:, 0, :, -1]

print('Done.')
