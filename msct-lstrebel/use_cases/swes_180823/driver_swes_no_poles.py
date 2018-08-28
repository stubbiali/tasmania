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
Driver for the Lax-Wenfroff finite-difference solver for the 
shallow water equations defined over a sphere.
"""
import numpy as np

import gridtools as gt
from swes_no_poles import LaxWendroffSWES

# Suggested values for $\alpha$ for first and second
# test cases from Williamson's suite:
# * 0
# * 0.05
# * pi/2 - 0.05
# * pi/2
ic = (0, 0)
#ic = (2, 0)

# Suggested simulation's length for Williamson's test cases:
# * IC 0: 12 days
# * IC 1: 14 days
t_final = 12

# Let's go!
solver = LaxWendroffSWES(planet=0, t_final=t_final, m=180, n=90, ic=ic,
						 cfl=1, diff=True, backend=gt.mode.NUMPY, dtype=np.float64)
t, phi, theta, h, u, v = solver.solve(verbose=100, save=100)

# Save data
filename = 'swes_no_poles_ic{}.npz'.format(ic[0])
np.savez(filename, t=t, phi=phi, theta=theta, h=h, u=u, v=v)
