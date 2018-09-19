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
import argparse

import gridtools as gt
from swes_no_poles import LaxWendroffSWES


parser = argparse.ArgumentParser(description="Run the Shallow Water Equation on a Sphere")
parser.add_argument("-nx", default=360, type=int,
					help="Number of grid points in x direction.")
parser.add_argument("-ny", default=180, type=int,
					help="Number of grid points in y direction.")
parser.add_argument("-ic", default=0, type=int,
					help="Initial condition either 0 or 2.")
parser.add_argument("-nt", default=12, type=int,
					help="Number of days the simulation should run.")
parser.add_argument("-sf", default=100, type=int,
					help="Save frequency: Number of time steps between fields are saved to file.")
args = parser.parse_args()

nx = args.nx
ny = args.ny
aic = args.ic
days = args.nt
sf = args.sf


# Suggested values for $\alpha$ for first and second
# test cases from Williamson's suite:
# * 0
# * 0.05
# * pi/2 - 0.05
# * pi/2
# ic = (0, 0)
ic = (aic, 0)

# Suggested simulation's length for Williamson's test cases:
# * IC 0: 12 days
# * IC 1: 14 days
t_final = days

# Let's go!
solver = LaxWendroffSWES(planet=0, t_final=t_final, m=nx, n=ny, ic=ic,
						 cfl=1, diff=True, backend=gt.mode.NUMPY, dtype=np.float64)
t, phi, theta, h, u, v = solver.solve(verbose=100, save=sf)

# Save data
filename = 'swes_no_poles_ic{}.npz'.format(ic[0])
np.savez(filename, t=t, phi=phi, theta=theta, h=h, u=u, v=v)


