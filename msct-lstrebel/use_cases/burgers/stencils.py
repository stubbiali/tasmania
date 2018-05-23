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
This module contains the definitions of some GT4Py stencils implementing
different schemes to solve the system of two-dimensional viscid Burgers' equations.
"""
import gridtools as gt

def stencil_burgers_forward_backward(dt, dx, dy, eps, in_u, in_v):
	"""
	GT4Py stencil solving the system of two-dimensional viscid Burgers' equations by relying on:
	
	* forward finite differences to discretize the time derivatives;
	* backward finite differences to discretize the first-order space derivatives;
	* a second-order centered scheme to discretize the second-order space derivatives.

	Arguments
	---------
	dt : GT4Py Global
		The timestep.
	dx : GT4Py Global
		The grid size in :math:`x`-direction.
	dy : GT4Py Global
		The grid size in :math:`y`-direction.
	eps : GT4Py Global
		The diffusion coefficient.
	in_u : GT4Py Equation
		The current :math:`x`-velocity.
	in_v : GT4Py Equation
		The current :math:`y`-velocity.

	Returns
	-------
	out_u : GT4Py Equation
		The :math:`x`-velocity at the next time level.
	out_v : GT4Py Equation
		The :math:`y`-velocity at the next time level.
	"""
	# Declare the indeces
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	# Instantiate the temporary and output fields
	lap_u = gt.Equation()
	lap_v = gt.Equation()
	out_u = gt.Equation()
	out_v = gt.Equation()

	# Compute the Laplacians
	lap_u[i, j, k] = (in_u[i-1, j, k] - 2. * in_u[i, j, k] + in_u[i+1, j, k]) / (dx ** 2) + \
					 (in_u[i, j-1, k] - 2. * in_u[i, j, k] + in_u[i, j+1, k]) / (dy ** 2)
	lap_v[i, j, k] = (in_v[i-1, j, k] - 2. * in_v[i, j, k] + in_v[i+1, j, k]) / (dx ** 2) + \
					 (in_v[i, j-1, k] - 2. * in_v[i, j, k] + in_v[i, j+1, k]) / (dy ** 2)

	# Step the solution
	out_u[i, j, k] = in_u[i, j, k] + dt * (- in_u[i, j, k] * (in_u[i, j, k] - in_u[i-1, j, k]) / dx \
										   - in_v[i, j, k] * (in_u[i, j, k] - in_u[i, j-1, k]) / dy \
										   + eps * lap_u[i, j, k])
	out_v[i, j, k] = in_v[i, j, k] + dt * (- in_u[i, j, k] * (in_v[i, j, k] - in_v[i-1, j, k]) / dx \
										   - in_v[i, j, k] * (in_v[i, j, k] - in_v[i, j-1, k]) / dy \
										   + eps * lap_v[i, j, k])

	return out_u, out_v

def stencil_burgers_upwind(dt, dx, dy, eps, in_u, in_v):
	"""
	GT4Py stencil solving the system of two-dimensional viscid Burgers' equations by relying on:
	
	* forward finite differences to discretize the time derivatives;
	* the upwind scheme to discretize the first-order space derivatives;
	* a second-order centered scheme to discretize the second-order space derivatives.

	Arguments
	---------
	dt : GT4Py Global
		The timestep.
	dx : GT4Py Global
		The grid size in :math:`x`-direction.
	dy : GT4Py Global
		The grid size in :math:`y`-direction.
	eps : GT4Py Global
		The diffusion coefficient.
	in_u : GT4Py Equation
		The current :math:`x`-velocity.
	in_v : GT4Py Equation
		The current :math:`y`-velocity.

	Returns
	-------
	out_u : GT4Py Equation
		The :math:`x`-velocity at the next time level.
	out_v : GT4Py Equation
		The :math:`y`-velocity at the next time level.
	"""
	# Declare the indeces
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	# Instantiate the temporary and output fields
	dudx  = gt.Equation()
	dudy  = gt.Equation()
	dvdx  = gt.Equation()
	dvdy  = gt.Equation()
	lap_u = gt.Equation()
	lap_v = gt.Equation()
	out_u = gt.Equation()
	out_v = gt.Equation()

	# Discretize the space derivatives of the fluxes via upwind
	dudx[i, j, k] = ((in_u[i, j, k] > 0.) * (in_u[i, j, k] - in_u[i-1, j, k]) +
					 (in_u[i, j, k] < 0.) * (in_u[i+1, j, k] - in_u[i, j, k])) / dx
	dudy[i, j, k] = ((in_v[i, j, k] > 0.) * (in_u[i, j, k] - in_u[i, j-1, k]) +
					 (in_v[i, j, k] < 0.) * (in_u[i, j+1, k] - in_u[i, j, k])) / dy
	dvdx[i, j, k] = ((in_u[i, j, k] > 0.) * (in_v[i, j, k] - in_v[i-1, j, k]) +
					 (in_u[i, j, k] < 0.) * (in_v[i+1, j, k] - in_v[i, j, k])) / dx
	dvdy[i, j, k] = ((in_v[i, j, k] > 0.) * (in_v[i, j, k] - in_v[i, j-1, k]) +
					 (in_v[i, j, k] < 0.) * (in_v[i, j+1, k] - in_v[i, j, k])) / dy

	# Compute the Laplacians
	lap_u[i, j, k] = (in_u[i-1, j, k] - 2. * in_u[i, j, k] + in_u[i+1, j, k]) / (dx ** 2) + \
					 (in_u[i, j-1, k] - 2. * in_u[i, j, k] + in_u[i, j+1, k]) / (dy ** 2)
	lap_v[i, j, k] = (in_v[i-1, j, k] - 2. * in_v[i, j, k] + in_v[i+1, j, k]) / (dx ** 2) + \
					 (in_v[i, j-1, k] - 2. * in_v[i, j, k] + in_v[i, j+1, k]) / (dy ** 2)

	# Step the solution
	out_u[i, j, k] = in_u[i, j, k] + dt * (- in_u[i, j, k] * dudx[i, j, k] \
										   - in_v[i, j, k] * dudy[i, j, k] \
										   + eps * lap_u[i, j, k])
	out_v[i, j, k] = in_v[i, j, k] + dt * (- in_u[i, j, k] * dvdx[i, j, k] \
										   - in_v[i, j, k] * dvdy[i, j, k] \
										   + eps * lap_v[i, j, k])

	return out_u, out_v

def stencil_burgers_upwind_third_order(dt, dx, dy, eps, in_u, in_v):
	"""
	GT4Py stencil solving the system of two-dimensional viscid Burgers' equations by relying on:
	
	* forward finite differences to discretize the time derivatives;
	* the third-order upwind scheme to discretize the first-order space derivatives;
	* a fourth-order centered scheme to discretize the second-order space derivatives.

	References
	----------
	Bethancourt A., and S. Komurasaki. *Solution of the 2D inviscid Burgers' equation \
		using a multi-directional upwind scheme*. Lecture notes.

	Arguments
	---------
	dt : GT4Py Global
		The timestep.
	dx : GT4Py Global
		The grid size in :math:`x`-direction.
	dy : GT4Py Global
		The grid size in :math:`y`-direction.
	eps : GT4Py Global
		The diffusion coefficient.
	in_u : GT4Py Equation
		The current :math:`x`-velocity.
	in_v : GT4Py Equation
		The current :math:`y`-velocity.

	Returns
	-------
	out_u : GT4Py Equation
		The :math:`x`-velocity at the next time level.
	out_v : GT4Py Equation
		The :math:`y`-velocity at the next time level.
	"""
	# Declare the indeces
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	# Instantiate the temporary and output fields
	dudx  = gt.Equation()
	dudy  = gt.Equation()
	dvdx  = gt.Equation()
	dvdy  = gt.Equation()
	lap_u = gt.Equation()
	lap_v = gt.Equation()
	out_u = gt.Equation()
	out_v = gt.Equation()

	# Discretize the space derivatives of the fluxes via third-order upwind
	dudx[i, j, k] = ((in_u[i, j, k] > 0.) * (+ 2. * in_u[i+1, j, k]
											 + 3. * in_u[  i, j, k]
											 - 6. * in_u[i-1, j, k]
											 +      in_u[i-2, j, k]) +
					 (in_u[i, j, k] < 0.) * (-      in_u[i+2, j, k] 
					 						 + 6. * in_u[i+1, j, k]
											 - 3. * in_u[  i, j, k]
											 - 2. * in_u[i-1, j, k])) / (6. * dx)
	dudy[i, j, k] = ((in_v[i, j, k] > 0.) * (+ 2. * in_u[i, j+1, k]
											 + 3. * in_u[i,   j, k]
											 - 6. * in_u[i, j-1, k]
											 +      in_u[i, j-1, k]) +
					 (in_v[i, j, k] < 0.) * (-      in_u[i, j+2, k] 
					 						 + 6. * in_u[i, j+1, k]
											 - 3. * in_u[i,   j, k]
											 - 2. * in_u[i, j-1, k])) / (6. * dy)
	dvdx[i, j, k] = ((in_u[i, j, k] > 0.) * (+ 2. * in_v[i+1, j, k]
											 + 3. * in_v[  i, j, k]
											 - 6. * in_v[i-1, j, k]
											 +      in_v[i-2, j, k]) +
					 (in_u[i, j, k] < 0.) * (-      in_v[i+2, j, k] 
					 						 + 6. * in_v[i+1, j, k]
											 - 3. * in_v[  i, j, k]
											 - 2. * in_v[i-1, j, k])) / (6. * dx)
	dvdy[i, j, k] = ((in_v[i, j, k] > 0.) * (+ 2. * in_v[i, j+1, k]
											 + 3. * in_v[i,   j, k]
											 - 6. * in_v[i, j-1, k]
											 +      in_v[i, j-1, k]) +
					 (in_v[i, j, k] < 0.) * (-      in_v[i, j+2, k] 
					 						 + 6. * in_v[i, j+1, k]
											 - 3. * in_v[i,   j, k]
											 - 2. * in_v[i, j-1, k])) / (6. * dy)

	# Compute the Laplacians
	lap_u[i, j, k] = (- 1./12. * in_u[i-2, j, k] 
					  + 4./3.  * in_u[i-1, j, k] 
					  - 5./2.  * in_u[  i, j, k]
					  + 4./3.  * in_u[i+1, j, k]
					  - 1./12. * in_u[i+2, j, k]) / (dx ** 2) + \
					 (- 1./12. * in_u[i, j-2, k] 
					  + 4./3.  * in_u[i, j-1, k] 
					  - 5./2.  * in_u[i,   j, k]
					  + 4./3.  * in_u[i, j+1, k]
					  - 1./12. * in_u[i, j+2, k]) / (dy ** 2)
	lap_v[i, j, k] = (- 1./12. * in_v[i-2, j, k] 
					  + 4./3.  * in_v[i-1, j, k] 
					  - 5./2.  * in_v[  i, j, k]
					  + 4./3.  * in_v[i+1, j, k]
					  - 1./12. * in_v[i+2, j, k]) / (dx ** 2) + \
					 (- 1./12. * in_v[i, j-2, k] 
					  + 4./3.  * in_v[i, j-1, k] 
					  - 5./2.  * in_v[i,   j, k]
					  + 4./3.  * in_v[i, j+1, k]
					  - 1./12. * in_v[i, j+2, k]) / (dy ** 2)

	# Step the solution
	out_u[i, j, k] = in_u[i, j, k] + dt * (- in_u[i, j, k] * dudx[i, j, k] \
										   - in_v[i, j, k] * dudy[i, j, k] \
										   + eps * lap_u[i, j, k])
	out_v[i, j, k] = in_v[i, j, k] + dt * (- in_u[i, j, k] * dvdx[i, j, k] \
										   - in_v[i, j, k] * dvdy[i, j, k] \
										   + eps * lap_v[i, j, k])

	return out_u, out_v
