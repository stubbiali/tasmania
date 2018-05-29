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
The system is subject to the initial and boundary conditions by Zhao et al.

References
----------
Zhao G., X. Yu, and R. Zhang. (2011). *The new numerical method for solving the system of \
	two-dimensional Burgers' equations*. Computers and Mathematics with Applications, 62:3279-3291.
"""
from datetime import datetime, timedelta
import numpy as np
import pickle

import gridtools as gt
import stencils

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
domain 		= [(0,0), (1,1)]		
nx     		= 101				
ny     		= 101			
dt     		= 0.001
nt     		= 1000	
eps    		= 0.01
method 		= 'upwind_third_order'
datatype	= np.float64	
save_freq	= 10			
print_freq	= 10		
filename	= 'test_zhao_' + str(method) + '.pickle'

#
# Driver
#
# Infer the grid size
dx = float(domain[1][0] - domain[0][0]) / (nx - 1)
dy = float(domain[1][1] - domain[0][1]) / (ny - 1)

# Create the grid
x  = np.linspace(domain[0][0], domain[1][0], nx)
xv = np.repeat(x[:, np.newaxis], ny, axis = 1)
y  = np.linspace(domain[0][1], domain[1][1], ny)
yv = np.repeat(y[np.newaxis, :], nx, axis = 0)

# Instatiate the arrays representing the solution
unow = np.zeros((nx, ny, 1), dtype = datatype)
unew = np.zeros((nx, ny, 1), dtype = datatype)
vnow = np.zeros((nx, ny, 1), dtype = datatype)
vnew = np.zeros((nx, ny, 1), dtype = datatype)

# Set the initial conditions
unew[:,:,0] = - 4. * eps * np.pi * np.cos(2 * np.pi * xv) * np.sin(np.pi * yv) / \
			(2. + np.sin(2. * np.pi * xv) * np.sin(np.pi * yv))
vnew[:,:,0] = - 2. * eps * np.pi * np.sin(2 * np.pi * xv) * np.cos(np.pi * yv) / \
			(2. + np.sin(2. * np.pi * xv) * np.sin(np.pi * yv))
			
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
	nb = 2

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

# Time integration
tsave = [timedelta(0),]
usave = np.copy(unew)
vsave = np.copy(vnew)
for n in range(nt):
	# Advance the time levels
	unow[:,:,0], vnow[:,:,0] = unew[:,:,0], vnew[:,:,0]

	# Step the solution
	stencil.compute()

	# Set the boundaries
	t = (n + 1) * float(dt)
	unew[ :nb, :, 0] = - 2. * eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t) * np.sin(np.pi * yv[ :nb, :])
	unew[-nb:, :, 0] = - 2. * eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t) * np.sin(np.pi * yv[-nb:, :])
	unew[:,  :nb, 0] = 0.
	unew[:, -nb:, 0] = 0.
	vnew[ :nb, :, 0] = 0.
	vnew[-nb:, :, 0] = 0.
	vnew[:,  :nb, 0] = - eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t) * np.sin(2. * np.pi * xv[:,  :nb])
	vnew[:, -nb:, 0] =   eps * np.pi * np.exp(- 5. * np.pi * np.pi * eps * t) * np.sin(2. * np.pi * xv[:, -nb:])
 
	# Check point
	if (print_freq > 0) and (n % print_freq == 0):
		print('Step %5.i, u max = %5.5f, u min = %5.5f, v max = %5.5f, v min = %5.5f, ||u|| max = %12.12f'
			  % (n+1, unew.max(), unew.min(), vnew.max(), vnew.min(), np.sqrt(unew**2 + vnew**2).max()))

	# Save
	if ((save_freq > 0) and (n % save_freq == 0)) or (n+1 == nt):
		tsave.append(timedelta(seconds = t))
		usave = np.concatenate((usave, unew), axis = 2)
		vsave = np.concatenate((vsave, vnew), axis = 2)

# Dump solution to a binary file
with open(filename, 'wb') as outfile:
	pickle.dump(tsave, outfile)
	pickle.dump(x, outfile)
	pickle.dump(y, outfile)
	pickle.dump(usave, outfile)
	pickle.dump(vsave, outfile)
	pickle.dump(eps, outfile)
