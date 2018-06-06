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
Script to simulate the system of two-dimensional viscid Burgers' equations.
The system is subject to the initial and boundary conditions by Shankar.

References
----------
https://ch.mathworks.com/matlabcentral/fileexchange/38087-burgers-equation-in-1d-and-2d.
"""
from datetime import datetime, timedelta
import numpy as np
import pickle

import gridtools as gt
import stencils
import timer

#
# User-defined settings:
# domain			South-west and north-east corners of the rectangular domain
# nx				Number of grid points in x-direction
# ny				Number of grid points in y-direction
# dt				Timestep [s]
# nt				Number of integration steps
# eps				Diffusion coefficient
# method			Method to use; options: 'forward_backward', 'upwind', 'upwind_third_order'
# datatype			Data type to use in computations
# save_freq			Save frequency; if 1, the solution is saved at each iteration
# print_freq		Print frequency; if 1, info about the solution are printed at each iteration
# filename			Path to the location where the solution will be saved
#

timer = timer.Timings(name="Burger's equation - Shankar setup")
timer.start(name="Overall Shankar time", level=1)
timer.start(name="Initialization", level=2)

domain 		= [(0,0), (2,2)]		
nx     		= 161				
ny     		= 161			
dt     		= 0.001	
nt     		= 600
eps    		= 0.01	
method 		= 'upwind_third_order'
datatype	= np.float64	
save_freq	= 10
print_freq	= 10
file_output     = False
filename	= 'test_shankar_' + str(method) + '.pickle'

#
# Driver
#
# Infer the grid size
dx = float(domain[1][0] - domain[0][0]) / (nx - 1)
dy = float(domain[1][1] - domain[0][1]) / (ny - 1)

# Create the grid
x = np.linspace(domain[0][0], domain[1][0], nx)
y = np.linspace(domain[0][1], domain[1][1], ny)

# Instatiate the arrays representing the solution
unow = np.zeros((nx, ny, 1), dtype = datatype)
unew = np.zeros((nx, ny, 1), dtype = datatype)
vnow = np.zeros((nx, ny, 1), dtype = datatype)
vnew = np.zeros((nx, ny, 1), dtype = datatype)

# Set the initial conditions
for i in range(nx):
	for j in range(ny):
		if (0.5 <= x[i] and x[i] <= 1.0) and (0.5 <= y[j] and y[j] <= 1.0):
			unew[i, j, 0], vnew[i, j, 0] = 0.0, 1.0
		else:
			unew[i, j, 0], vnew[i, j, 0] = 1.0, 0.0

# Apply the boundary conditions
unew[ 0,  :, 0], vnew[ 0,  :, 0] = 0., 0.
unew[-1,  :, 0], vnew[-1,  :, 0] = 0., 0.
unew[ :,  0, 0], vnew[ :,  0, 0] = 0., 0.
unew[ :, -1, 0], vnew[ :, -1, 0] = 0., 0.

# Set stencil's definitions function and computational domain
if method == 'forward_backward':
	definitions_func_ = stencils.stencil_burgers_forward_backward
	domain_ = gt.domain.Rectangle((1, 1, 0), (nx-2, ny-2, 0))
	nb = 1
elif method == 'upwind':
	definitions_func_ = stencils.stencil_burgers_upwind
	domain_ = gt.domain.Rectangle((1, 1, 0), (nx-2, ny-2, 0))
	nb = 1
elif method == 'upwind_third_order':
	definitions_func_ = stencils.stencil_burgers_upwind_third_order
	domain_ = gt.domain.Rectangle((2, 2, 0), (nx-3, ny-3, 0))
	nb = 1

# Convert global inputs to GT4Py Global's
dt_  = gt.Global(dt)
dx_  = gt.Global(dx)
dy_  = gt.Global(dy)
eps_ = gt.Global(eps)

# Instantiate stencil object
stencil = gt.NGStencil(
	definitions_func = definitions_func_,
	inputs = {'in_u': unow, 'in_v': vnow},
	global_inputs = {'dt': dt_, 'dx': dx_, 'dy': dy_, 'eps': eps_},
	outputs = {'out_u': unew, 'out_v': vnew},
	domain = domain_,
	mode = gt.mode.NUMPY)

timer.stop(name="Initialization")
timer.start(name="Time integration", level=2)

# Time integration
tsave = [timedelta(0),]
usave = np.copy(unew)
vsave = np.copy(vnew)
for n in range(nt):
	# Advance the time levels
	unow[:,:,0], vnow[:,:,0] = unew[:,:,0], vnew[:,:,0]

	# Step the solution
	stencil.compute()

	# Apply the boundary conditions
	unew[  1:nb,   1:-1, 0], vnew[  1:nb,   1:-1, 0] = unow[  1:nb,   1:-1, 0], vnow[  1:nb,   1:-1, 0]
	unew[-nb:-1,   1:-1, 0], vnew[-nb:-1,   1:-1, 0] = unow[-nb:-1,   1:-1, 0], vnow[-nb:-1,   1:-1, 0]
	unew[  1:-1,   1:nb, 0], vnew[  1:-1,   1:nb, 0] = unow[  1:-1,   1:nb, 0], vnow[  1:-1,   1:nb, 0]
	unew[  1:-1, -nb:-1, 0], vnew[  -nb:, -nb:-1, 0] = unow[  1:-1, -nb:-1, 0], vnow[  -nb:, -nb:-1, 0]

	timer.start(name="Checkpointing during time integration", level=3)

	# Check point
	if (print_freq > 0) and (n % print_freq == 0):
		print('Step %5.i, u max = %5.5f, u min = %5.5f, v max = %5.5f, v min = %5.5f, ||u|| max = %12.12f'
			  % (n+1, unew.max(), unew.min(), vnew.max(), vnew.min(), np.sqrt(unew**2 + vnew**2).max()))

	# Save
	if ((save_freq > 0) and (n % save_freq == 0)) or (n+1 == nt):
		tsave.append(timedelta(seconds = (n+1) * dt))
		usave = np.concatenate((usave, unew), axis = 2)
		vsave = np.concatenate((vsave, vnew), axis = 2)

	timer.stop(name="Checkpointing during time integration")

timer.stop(name="Time integration")

if file_output:
	timer.start(name="Writing to file", level=2)

	# Dump solution to a binary file
	with open(filename, 'wb') as outfile:
		pickle.dump(tsave, outfile)
		pickle.dump(x, outfile)
		pickle.dump(y, outfile)
		pickle.dump(usave, outfile)
		pickle.dump(vsave, outfile)
		pickle.dump(eps, outfile)

	timer.stop(name="Writing to file")

timer.stop(name="Overall Shankar time")

timer.list_timings()


