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
## @package gt4ess
#  Test PotentialTemperature2d class.

import numpy as np

from grids.potential_temperature import PotentialTemperature2d as Grid

# Set domain and number of grid points
domain_x = (0., 100.)
n_x = 1001
domain_theta = (5., 10.)
n_theta = 51

# Specify terrain surface profile
h_s = '500 * exp((- x*x / (10.*10.)))'

# Instantiate grid
grid = Grid(domain_x, n_x, domain_theta, n_theta, h_s)