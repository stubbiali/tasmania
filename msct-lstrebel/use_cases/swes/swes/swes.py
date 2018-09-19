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
import copy
import math
import pickle
import sys
import time

import numpy as np
import gridtools as gt
import timer as ti


							 
class LaxWendroffSWES:
	"""
	Implementation of the finite difference Lax-Wendroff scheme
	for the shallow water equations defined on a sphere.
	"""
	def __init__(self, planet, t_final, m, n, ic, cfl, diff, backend,
				 dtype=np.float64):
		"""
		Constructor.

		Parameters
		----------
		planet : int
			Integer denoting the planet on which set the equations.
			Available options are:

				* 0, for the Earth;
				* 1, for Saturn.

		t_final : float
			Simulation length in days.
		m : int
			Number of grid points in the longitude direction.
		n : int
			Number of grid points in the latitude direction.
		ic : tuple
			Tuple storing the identifier of the initial condition,
			followed by optional parameters. Available options for
			the identifier are:

				* 0, for the test case #1 by Williamson et al.;
				* 1, for the test case #2 by Williamson et al.;
				* 2, for the test case #6 by Williamson et al..

		cfl : float
			CFL number.
		diff : bool
			:obj:`True` to switch on numerical diffusion, :obj:`False` otherwise.
		backend : obj
			GT4Py backend.
		dtype : `obj`, float
			Type to be used for all NumPy arrays used with this object.
			Defaults tp :obj:`numpy.float64`.
		"""
		# Set planet constants
		# For Saturn, all the values are retrieved from:
		# http://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html
		self.timer = ti.Timings(name="Shallow Water Equation on a Sphere")
		self.timer.start(name="Overall SWES time", level=1)
		self.timer.start(name="Initialization", level=2)
		self.planet = planet
		if self.planet == 0:  # Earth
			self.g				= 9.80616
			self.a				= 6.37122e6
			self.omega			= 7.292e-5
			self.scale_height	= 8e3
			self.nu 			= 5e5
		elif self.planet == 1:  # Saturn
			self.g				= 10.44
			self.a				= 5.8232e7
			self.omega			= 2. * math.pi / (10.656 * 3600.)
			self.scale_height	= 60e3
			self.nu 			= 5e6
		else:
			raise ValueError('Unknown planet {}.'.format(planet))
						
		# 
		# Lat-lon grid
		#
		assert (m > 1) and (n > 1), \
			"Number of grid points along each direction must be greater than one."
		# Discretize longitude
		self.m = m			
		self.dphi = 2. * math.pi / self.m
		self.phi_1d = np.linspace(-self.dphi, 2.*math.pi + self.dphi, self.m+3,
								  dtype=dtype)

		# Discretize latitude
		self.n = n
		self.dtheta = math.pi / (self.n - 1)
		self.theta_1d = np.linspace(-0.5 * math.pi, 0.5 * math.pi, self.n,
									dtype=dtype)
		
		# Build grid
		self.phi, self.theta = np.meshgrid(self.phi_1d, self.theta_1d, indexing='ij')
		
		# Compute cos(theta)
		self.c      = np.cos(self.theta)
		self.c_midy = np.cos(0.5 * (self.theta[:, :-1] + self.theta[:, 1:]))
				
		# Compute tan(theta)
		self.tg      = np.tan(self.theta)
		self.tg_midx = np.tan(0.5 * (self.theta[:-1, :] + self.theta[1:, :]))
		self.tg_midy = np.tan(0.5 * (self.theta[:, :-1] + self.theta[:, 1:]))
				
		# Coriolis term
		self.f = 2. * self.omega * np.sin(self.theta)

		# 
		# Cartesian coordinates and increments
		# 
		self.x	= self.a * np.cos(self.theta) * self.phi
		self.y	= self.a * self.theta
		self.y1 = self.a * np.sin(self.theta)
		
		self.dx  = self.x[1:, :] - self.x[:-1, :]
		self.dy  = self.y[:, 1:] - self.y[:, :-1]
		self.dy1 = self.y1[:, 1:] - self.y1[:, :-1]
		
		# Compute minimum longitudinal and latitudinal distance between
		# adjacent grid points, needed to compute timestep size through
		# CFL condition
		self.dx_min = self.dx[:, 1:-1].min()
		self.dy_min = self.dy.min()
		
		# "Centred" increments
		self.dxc  = np.zeros((self.m+3, self.n), dtype=dtype)
		self.dxc[1:-1, 1:-1]  = 0.5 * (self.dx[:-1, 1:-1] + self.dx[1:, 1:-1])
		self.dyc  = np.zeros((self.m+3, self.n), dtype=dtype)
		self.dyc[1:-1, 1:-1]  = 0.5 * (self.dy[1:-1, :-1] + self.dy[1:-1, 1:])
		self.dy1c = np.zeros((self.m+3, self.n), dtype=dtype)
		self.dy1c[1:-1, 1:-1] = 0.5 * (self.dy1[1:-1, :-1] + self.dy1[1:-1, 1:])
		
		# Compute longitudinal increment used in pole treatment
		self.dxp = 2. * self.a * np.sin(self.dtheta) / \
				   (1. + np.cos(self.dtheta))
								  
		# Compute map factor at the poles
		self.m_north = 2. / (1. + np.sin(self.theta_1d[-2]))
		self.m_south = 2. / (1. - np.sin(self.theta_1d[1]))
				
		# 
		# Time discretization
		# 
		assert(t_final >= 0.), "Final time must be non-negative."
		
		# Convert simulation length from days to seconds
		self.t_final = 24. * 3600. * t_final
		
		# Set maximum number of iterations
		if self.t_final == 0.:
			self.nmax = 50
			self.t_final = 3600.
		else:
			self.nmax = sys.maxsize
		
		# CFL number
		self.cfl = cfl

		# Initialize timestep size
		self.dt = gt.Global(0.)
				
		# 
		# Flat terrain height
		#
		self.hs = np.zeros((self.m+3, self.n), dtype=dtype)
		
		# 
		# Numerical settings
		#
		assert ic[0] in range(3), 'Invalid problem identifier. ' \
								  'See code documentation for supported initial conditions.'
		
		self.ic = ic
		self.diffusion = diff
		
		if self.ic[0] == 0:
			self.only_advection = True
		else:
			self.only_advection = False

		self.backend = backend
		
		# 
		# Pre-compute coefficients for second-order three
		# points approximations of first-order derivative 
		# 
		if self.diffusion:
			# Centred finite difference along longitude
			# Ax, Bx and Cx denote the coefficients associated 
			# with the centred, downwind and upwind point, respectively
			self.Ax = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.Ax[2:-2, 2:-2] = (self.dx[1:, 1:-1] - self.dx[:-1, 1:-1]) / \
								  (self.dx[1:, 1:-1] * self.dx[:-1, 1:-1])
			self.Ax[1, :], self.Ax[-2, :] = self.Ax[-4, :], self.Ax[3, :]
			
			self.Bx = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.Bx[2:-2, 2:-2] = self.dx[:-1, 1:-1] / \
								  (self.dx[1:, 1:-1] * (self.dx[1:, 1:-1] + self.dx[:-1, 1:-1]))
			self.Bx[1, :], self.Bx[-2, :] = self.Bx[-4, :], self.Bx[3, :]
			
			self.Cx = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.Cx[2:-2, 2:-2] = - self.dx[1:, 1:-1] / \
								  (self.dx[:-1, 1:-1] * (self.dx[1:, 1:-1] + self.dx[:-1, 1:-1]))
			self.Cx[1, :], self.Cx[-2, :] = self.Cx[-4, :], self.Cx[3, :]
		
			# Centred finite difference along latitude
			# Ay, By and Cy denote the coefficients associated 
			# with the centred, downwind and upwind point, respectively
			self.Ay = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.Ay[2:-2, 2:-2] = (self.dy[1:-1, 1:] - self.dy[1:-1, :-1]) / \
								  (self.dy[1:-1, 1:] * self.dy[1:-1, :-1])
						
			self.By = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.By[2:-2, 2:-2] = self.dy[1:-1, :-1] / \
								  (self.dy[1:-1, 1:] * (self.dy[1:-1, 1:] + self.dy[1:-1, :-1]))
			self.By[2:-2, -2] = 1. / (2. * self.dy[1:-1, -2])
			self.By[2:-2, 1] = 1. / (2. * self.dy[1:-1, 1])
			
			self.Cy = np.zeros((self.m+5, self.n+2), dtype=dtype)
			self.Cy[2:-2, 2:-2] = - self.dy[1:-1, 1:] / \
								  (self.dy[1:-1, :-1] * (self.dy[1:-1, 1:] + self.dy[1:-1, :-1]))
			self.Cy[2:-2, -2] = - 1. / (2. * self.dy[1:-1, -2])
			self.Cy[2:-2, 1] = - 1. / (2. * self.dy[1:-1, 1])
						  
		# 
		# Set initial conditions
		# 
		self._set_initial_conditions(dtype)

		self.timer.stop(name="Initialization")
										
	def _set_initial_conditions(self, dtype):
		"""
		Set the initial conditions.
		"""
		# 
		# First test case taken from Williamson's suite 
		#
		if self.ic[0] == 0:
			# Extract alpha
			alpha = self.ic[1]
			
			# Define coefficients
			u0 = 2. * math.pi * self.a / (12. * 24. * 3600.)
			h0 = 1000.
			phi_c = 1.5 * math.pi
			theta_c = 0.
			R = self.a / 3.
			
			# Compute steering wind
			self.u = u0 * (np.cos(self.theta) * np.cos(alpha) +
			          	   np.sin(self.theta) * np.cos(self.phi) * np.sin(alpha))
			self.u_midx = np.zeros((self.m+2, self.n), dtype=dtype)
			self.u_midx[:, 1:-1] = u0 * (np.cos(0.5 * (self.theta[:-1, 1:-1] +
													   self.theta[1:, 1:-1])) * np.cos(alpha) +
						  	   	  		 np.sin(0.5 * (self.theta[:-1, 1:-1] +
													   self.theta[1:, 1:-1])) *
						  	   	  		 np.cos(0.5 * (self.phi[:-1, 1:-1] +
													   self.phi[1:, 1:-1])) * np.sin(alpha))
						  	   
			self.v = - u0 * np.sin(self.phi) * np.sin(alpha)
			self.v_midy = np.zeros((self.m+3, self.n-1), dtype=dtype)
			self.v_midy[1:-1, :] = - u0 * np.sin(0.5 * (self.phi[1:-1, :-1] +
														self.phi[1:-1, 1:])) * np.sin(alpha)
			
			# Compute initial fluid height
			r = self.a * np.arccos(np.sin(theta_c) * np.sin(self.theta) +
								   np.cos(theta_c) * np.cos(self.theta) *
								   np.cos(self.phi - phi_c))
			self.h = np.where(r < R, 0.5 * h0 * (1. + np.cos(math.pi * r / R)), 0.)
						
		# 
		# Second test case taken from Williamson's suite 
		#
		if self.ic[0] == 1:
			# Extract alpha
			alpha = self.ic[1]
		
			# Set constants
			u0 = 2. * math.pi * self.a / (12. * 24. * 3600.)
			h0 = 2.94e4 / self.g
			
			# Make Coriolis parameter dependent on longitude and latitude
			self.f = 2. * self.omega * \
					 (- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) +
					  np.sin(self.theta) * np.cos(alpha))
					  
			# Compute initial height
			self.h = h0 - (self.a * self.omega * u0 + 0.5 * (u0 ** 2.)) * \
					 ((- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) +
					   np.sin(self.theta) * np.cos(alpha)) ** 2.)
					   
			# Compute initial wind
			self.u = u0 * (np.cos(self.theta) * np.cos(alpha) +
					  	   np.cos(self.phi) * np.sin(self.theta) * np.sin(alpha))
			self.v = - u0 * np.sin(self.phi) * np.sin(alpha)

		#
		# Sixth test case taken from Williamson's suite
		# 		
		if self.ic[0] == 2:
			# Set constants
			w  = 7.848e-6
			K  = 7.848e-6
			h0 = 8e3
			R  = 4.
			
			# Compute initial fluid height
			A = 0.5 * w * (2. * self.omega + w) * (np.cos(self.theta) ** 2.) + \
				0.25 * (K ** 2.) * (np.cos(self.theta) ** (2. * R)) * \
				((R + 1.) * (np.cos(self.theta) ** 2.) +
				 (2. * (R ** 2.) - R - 2.) -
				 2. * (R ** 2.) * (np.cos(self.theta) ** (-2.))) 
			B = (2. * (self.omega + w) * K) / ((R + 1.) * (R + 2.)) * \
				(np.cos(self.theta) ** R) * \
				(((R ** 2.) + 2. * R + 2.) -
				 ((R + 1.) ** 2.) * (np.cos(self.theta) ** 2.))
			C = 0.25 * (K ** 2.) * (np.cos(self.theta) ** (2. * R)) * \
				((R + 1.) * (np.cos(self.theta) ** 2.) - (R + 2.)) 
									  
			self.h = h0 + ((self.a ** 2.) * A +
					  	   (self.a ** 2.) * B * np.cos(R * self.phi) +
					  	   (self.a ** 2.) * C * np.cos(2. * R * self.phi)) / self.g
					 
			# Compute initial wind
			self.u = self.a * w * np.cos(self.theta) + \
					 self.a * K * (np.cos(self.theta) ** (R - 1.)) * \
					 (R * (np.sin(self.theta) ** 2.) - (np.cos(self.theta) ** 2.)) * \
					 np.cos(R * self.phi)
			self.v = - self.a * K * R * (np.cos(self.theta) ** (R - 1.)) * \
				  	 np.sin(self.theta) * np.sin(R * self.phi)  			

		# 
		# Set height at the poles 
		# If values at different longitudes are not\ the same, average them out
		self.h[:, -1] = np.sum(self.h[1:-2, -1]) / self.m
		self.h[:, 0]  = np.sum(self.h[1:-2, 0]) / self.m
		
		# 
		# Stereographic wind components at the poles
		# 
		if not self.only_advection:
			# Compute stereographic components at North pole
			# at each longitude and then make an average
			u_north = - self.u[1:-2, -1] * np.sin(self.phi_1d[1:-2]) \
					  - self.v[1:-2, -1] * np.cos(self.phi_1d[1:-2])
			self.u_north = np.sum(u_north) / self.m
		
			v_north = + self.u[1:-2, -1] * np.cos(self.phi_1d[1:-2]) \
					  - self.v[1:-2, -1] * np.sin(self.phi_1d[1:-2])
			self.v_north = np.sum(v_north) / self.m
					
			# Compute stereographic components at South pole
			# at each longitude and then make an average
			u_south = - self.u[1:-2, 0] * np.sin(self.phi_1d[1:-2]) \
					  + self.v[1:-2, 0] * np.cos(self.phi_1d[1:-2])
			self.u_south = np.sum(u_south) / self.m
		
			v_south = + self.u[1:-2, 0] * np.cos(self.phi_1d[1:-2]) \
					  + self.v[1:-2, 0] * np.sin(self.phi_1d[1:-2])
			self.v_south = np.sum(v_south) / self.m
		
	def _extend_solution(self, in_q, in_q_ext, tmp_q, tmp_q_ext):
		"""
		Extend the solution to accommodate for numerical diffusion.
		Periodic and cross-pole conditions are applied along longitude
		and latitude, respectively.
		"""
		# Extend current solution
		in_q_ext[1:-1, 1:-1] = in_q
		in_q_ext[0, 1:-1]    = in_q[-4, :]
		in_q_ext[-1, 1:-1]   = in_q[3, :]
		in_q_ext[2:-2, -1]   = np.concatenate((in_q[int((self.m+3)/2):-1, -2],
											   in_q[1:int((self.m+3)/2),  -2]), axis=0)
		in_q_ext[2:-2, 0]    = np.concatenate((in_q[int((self.m+3)/2):-1,  1],
											   in_q[1:int((self.m+3)/2),   1]), axis=0)
								
		# Extend provisional solution
		tmp_q_ext[1:-1, 1:-1] = tmp_q		

	@staticmethod
	def definitions_advection(dt, dx, dxc, dy1, dy1c, c, c_midy,
							  u, u_midx, v, v_midy, in_h):
		"""
		Definitions function for the GT4Py stencil integrating
		pure advection equations.
		"""
		#
		# Indices
		#
		i = gt.Index()
		j = gt.Index()

		#
		# Temporary and output arrays
		#
		h_midx = gt.Equation()
		h_midy = gt.Equation()
		out_h = gt.Equation()

		#
		# Computations
		#
		h_midx[i, j] = 0.5 * (in_h[i, j] + in_h[i+1, j]) - \
					   0.5 * dt / dx[i, j] * \
					   (in_h[i+1, j] * u[i+1, j] - in_h[i, j] * u[i, j])
		h_midy[i, j] = 0.5 * (in_h[i, j] + in_h[i, j+1]) - \
					   0.5 * dt / dy1[i, j] * \
					   (in_h[i, j+1] * v[i, j+1] * c[i, j+1] - 
					    in_h[i, j] * v[i, j] * c[i, j])
		out_h[i, j] = in_h[i, j] - \
					  dt / dxc[i, j] * \
					  (h_midx[i, j] * u_midx[i, j] - 
					   h_midx[i-1, j] * u_midx[i-1, j]) - \
					  dt / dy1c[i, j] * \
					  (h_midy[i, j] * v_midy[i, j] * c_midy[i, j] -
					   h_midy[i, j-1] * v_midy[i, j-1] * c_midy[i, j-1])
		
		return out_h

	def definitions_lax_wendroff(self, dt, dx, dxc, dy, dyc, dy1, dy1c, 
								 c, c_midy, f, hs, tg, tg_midx, tg_midy, 
								 in_h, in_u, in_v):
		"""
        Definitions function for the GT4Py stencil implementing the
        Lax-Wendroff scheme for the shallow water equations defined
        on a sphere.
        """
		#
		# Indices
		#
		i = gt.Index()
		j = gt.Index()

		#
		# Temporary and output arrays
		#
		v1 = gt.Equation()
		hu = gt.Equation()
		hv = gt.Equation()
		hv1 = gt.Equation()

		h_midx = gt.Equation()
		Ux = gt.Equation()
		hu_midx = gt.Equation()
		Vx = gt.Equation()
		hv_midx = gt.Equation()

		h_midy = gt.Equation()
		Uy = gt.Equation()
		hu_midy = gt.Equation()
		Vy1 = gt.Equation()
		Vy2 = gt.Equation()
		hv_midy = gt.Equation()

		out_h = gt.Equation()

		Ux_mid = gt.Equation()
		Uy_mid = gt.Equation()
		out_hu = gt.Equation()
		out_u = gt.Equation()

		Vx_mid = gt.Equation()
		Vy1_mid = gt.Equation()
		Vy2_mid = gt.Equation()
		out_hv = gt.Equation()	
		out_v = gt.Equation()
		
		#
		# Compute
		# 
		v1[i, j] = in_v[i, j] * c[i, j]
		hu[i, j] = in_h[i, j] * in_u[i, j]
		hv[i, j] = in_h[i, j] * in_v[i, j]
		hv1[i, j] = hv[i, j] * c[i, j]

		# Longitudinal mid-values for h
		h_midx[i, j] = 0.5 * (in_h[i, j] + in_h[i+1, j]) - \
			 		   0.5 * dt / dx[i, j] * (hu[i+1, j] - hu[i, j]) 

		# Longitudinal mid-values for hu 
		Ux[i, j] = hu[i, j] * in_u[i, j] + 0.5 * self.g * in_h[i, j] * in_h[i, j]
		hu_midx[i, j] = 0.5 * (hu[i, j] + hu[i+1, j]) - \
				  		0.5 * dt / dx[i, j] * (Ux[i+1, j] - Ux[i, j]) + \
				  		0.5 * dt * \
				  		(0.5 * (f[i, j] + f[i+1, j]) + 
				   		 0.5 * (in_u[i, j] + in_u[i+1, j]) * tg_midx[i, j] / self.a) * \
				  		0.5 * (hv[i, j] + hv[i+1, j])

		# Longitudinal mid-values for hv
		Vx[i, j] = hu[i, j] * in_v[i, j]
		hv_midx[i, j] = 0.5 * (hv[i, j] + hv[i+1, j]) - \
				  		0.5 * dt / dx[i, j] * (Vx[i+1, j] - Vx[i, j]) - \
				  		0.5 * dt * \
				  		(0.5 * (f[i, j] + f[i+1, j]) + 
				   		 0.5 * (in_u[i, j] + in_u[i+1, j]) * tg_midx[i, j] / self.a) * \
				  		0.5 * (hu[i, j] + hu[i+1, j])

		# Latitudinal mid-values for h
		h_midy[i, j] = 0.5 * (in_h[i, j] + in_h[i, j+1]) - \
			 		   0.5 * dt / dy1[i, j] * (hv1[i, j+1] - hv1[i, j])
			  
		# Latitudinal mid-values for hu
		Uy[i, j] = hu[i, j] * v1[i, j]
		hu_midy[i, j] = 0.5 * (hu[i, j] + hu[i, j+1]) - \
				  		0.5 * dt / dy1[i, j] * (Uy[i, j+1] - Uy[i, j]) + \
						0.5 * dt * \
				  		(0.5 * (f[i, j] + f[i, j+1]) + 
				   		 0.5 * (in_u[i, j] + in_u[i, j+1]) * tg_midy[i, j] / self.a) * \
				  		0.5 * (hv[i, j] + hv[i, j+1])
				  
		# Latitudinal mid-values for hv
		Vy1[i, j] = hv[i, j] * v1[i, j]
		Vy2[i, j] = 0.5 * self.g * in_h[i, j] * in_h[i, j]
		hv_midy[i, j] = 0.5 * (hv[i, j] + hv[i, j+1]) - \
			  			0.5 * dt / dy1[i, j] * (Vy1[i, j+1] - Vy1[i, j]) - \
						0.5 * dt / dy[i, j] * (Vy2[i, j+1] - Vy2[i, j]) - \
					    0.5 * dt * \
					    (0.5 * (f[i, j] + f[i, j+1]) + 
					     0.5 * (in_u[i, j] + in_u[i, j+1]) * tg_midy[i, j] / self.a) * \
					    0.5 * (hu[i, j] + hu[i, j+1])
		
		# Update h
		out_h[i, j] = in_h[i, j] - \
			   		  dt / dxc[i, j] * (hu_midx[i, j] - hu_midx[i-1, j]) - \
			   		  dt / dy1c[i, j] * (hv_midy[i, j] * c_midy[i, j] -
					  					 hv_midy[i, j-1] * c_midy[i, j-1])
			   
		# Update hu
		Ux_mid[i, j] = (h_midx[i, j] > 0.) * hu_midx[i, j] * hu_midx[i, j] / h_midx[i, j] + \
					   0.5 * self.g * h_midx[i, j] * h_midx[i, j]
		Uy_mid[i, j+0.5] = (h_midy[i, j] > 0.) * hv_midy[i, j] * c_midy[i, j] * \
						   hu_midy[i, j] / h_midy[i, j]
		out_hu[i, j] = hu[i, j] - \
					   dt / dxc[i, j] * (Ux_mid[i, j] - Ux_mid[i-1, j]) - \
					   dt / dy1c[i, j] * (Uy_mid[i, j] - Uy_mid[i, j-1]) + \
					   dt * (f[i, j] +
					 		 0.25 * (hu_midx[i-1, j] / h_midx[i-1, j] +
					 		   		 hu_midx[i, j] / h_midx[i, j] +
					 		   		 hu_midy[i, j-1] / h_midy[i, j-1] +
					 		   		 hu_midy[i, j] / h_midy[i, j]) *
					 		 tg[i, j] / self.a) * \
					   0.25 * (hv_midx[i-1, j] + hv_midx[i, j] +
					   		   hv_midy[i, j-1] + hv_midy[i, j]) - \
					   dt * self.g * \
					   0.25 * (h_midx[i-1, j] + h_midx[i, j] +
					   		   h_midy[i, j-1] + h_midy[i, j]) * \
					   (hs[i+1, j] - hs[i-1, j]) / (dx[i-1, j] + dx[i, j])
				   
		# Update hv
		Vx_mid[i, j] = (h_midx[i, j] > 0.) * hv_midx[i, j] * hu_midx[i, j] / h_midx[i, j]
		Vy1_mid[i, j] = (h_midy[i, j] > 0.) * hv_midy[i, j] * hv_midy[i, j] / \
						h_midy[i, j] * c_midy[i, j]
		Vy2_mid[i, j] = 0.5 * self.g * h_midy[i, j] * h_midy[i, j]
		out_hv[i, j] = hv[i, j] - \
					   dt / dxc[i, j] * (Vx_mid[i, j] - Vx_mid[i-1, j]) - \
					   dt / dy1c[i, j] * (Vy1_mid[i, j] - Vy1_mid[i, j-1]) - \
					   dt / dyc[i, j] * (Vy2_mid[i, j] - Vy2_mid[i, j-1]) - \
					   dt * (f[i, j] +
						     0.25 * (hu_midx[i-1, j] / h_midx[i-1, j] +
								     hu_midx[i, j] / h_midx[i, j] +
								     hu_midy[i, j-1] / h_midy[i, j-1] +
								     hu_midy[i, j] / h_midy[i, j]) *
						     tg[i, j] / self.a) * \
					   0.25 * (hu_midx[i-1, j] + hu_midx[i, j] +
					   		   hu_midy[i, j-1] + hu_midy[i, j]) - \
					   dt * self.g * \
					   0.25 * (h_midx[i-1, j] + h_midx[i, j] +
					   		   h_midy[i, j-1] + h_midy[i, j]) * \
					   (hs[i, j+1] - hs[i, j-1]) / (dy1[i, j-1] + dy1[i, j])
					
		# Come back to original variables
		out_u[i, j] = out_hu[i, j] / out_h[i, j]
		out_v[i, j] = out_hv[i, j] / out_h[i, j]

		return out_h, out_u, out_v

	def definitions_diffusion(self, dt, Ax, Ay, Bx, By, Cx, Cy, in_q, tmp_q):
		"""
        Definitions function for the GT4Py stencil applying numerical diffusion.
        """
		#
		# Indices
		#
		i = gt.Index()
		j = gt.Index()

		#
		# Temporary and output arrays
		#
		qxx = gt.Equation()
		qyy = gt.Equation()
		out_q = gt.Equation()

		#
		# Compute
		#
		qxx[i, j] = Ax[i, j] * (Ax[i, j] * in_q[i, j] +
								Bx[i, j] * in_q[i+1, j] +
								Cx[i, j] * in_q[i-1, j]) + \
			  		Bx[i, j] * (Ax[i+1, j] * in_q[i+1, j] +
			  					Bx[i+1, j] * in_q[i+2, j] +
			  					Cx[i+1, j] * in_q[i, j]) + \
			  		Cx[i, j] * (Ax[i-1, j] * in_q[i-1, j] +
			  					Bx[i-1, j] * in_q[i, j] +
			  					Cx[i-1, j] * in_q[i-2, j])
			  
		qyy[i, j] = Ay[i, j] * (Ay[i, j] * in_q[i, j] +
								By[i, j] * in_q[i, j+1] +
								Cy[i, j] * in_q[i, j-1]) + \
			  		By[i, j] * (Ay[i, j+1] * in_q[i, j+1] +
			  					By[i, j+1] * in_q[i, j+2] +
			  					Cy[i, j+1] * in_q[i, j]) + \
			  		Cy[i, j] * (Ay[i, j-1] * in_q[i, j-1] +
			  					By[i, j-1] * in_q[i, j] +
			  					Cy[i, j-1] * in_q[i, j-2])
			   
		out_q[i, j] = tmp_q[i, j] + dt * self.nu * (qxx[i, j] + qyy[i, j])
		
		return out_q

	def solve(self, verbose, save):
		"""
		Perform the time marching.

		Parameters
		----------
		verbose	: int
			If positive, print to screen information about the solution
			every :obj:`verbose` timesteps. If negative, print to screen
			information only about the initial and final state.
		save : int
			If positive, store the solution every :obj:`save` timesteps.
			If negative, store only the initial and final state.

		Returns
		-------
		t_save : array_like
			1-D :class:`numpy.ndarray` collecting the time instants (in seconds)
			at which solution has been stored.
		phi : array_like
			2-D :class:`numpy.ndarray` collecting the longitudinal coordinates
			of all grid points.
		theta : array_like
			2-D :class:`numpy.ndarray` collecting the latitudinal coordinates
			of all grid points.
		h_save : array_like
			3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
			fluid height at time :obj:`t_save[k]`.
		u_save : array_like
			3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
			u-velocity at time :obj:`t_save[k]`.
		v_save : array_like
			3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
			v-velocity at time :obj:`t_save[k]`.
		"""
		verbose = int(verbose)
		save = int(save)
		
		#
		# Allocate output arrays
		#
		dtype = self.theta.dtype
		h_new = np.zeros((self.m+3, self.n), dtype=dtype)
		u_new = np.zeros((self.m+3, self.n), dtype=dtype)
		v_new = np.zeros((self.m+3, self.n), dtype=dtype)
		if self.diffusion:
			h_ext 	  = np.zeros((self.m+5, self.n+2), dtype=dtype)
			u_ext 	  = np.zeros((self.m+5, self.n+2), dtype=dtype)
			v_ext 	  = np.zeros((self.m+5, self.n+2), dtype=dtype)
			h_tmp_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)
			u_tmp_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)
			v_tmp_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)
			h_new_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)
			u_new_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)
			v_new_ext = np.zeros((self.m+5, self.n+2), dtype=dtype)

		# 
		# Initialize stencils
		# 
		if self.only_advection:
			stencil = gt.NGStencil(
				definitions_func = self.definitions_advection,
				inputs 			 = {'in_h': self.h, 'dx': self.dx, 'dxc': self.dxc,
									'dy1': self.dy1, 'dy1c': self.dy1c,
									'c': self.c, 'c_midy': self.c_midy,
									'u': self.u, 'u_midx': self.u_midx,
									'v': self.v, 'v_midy': self.v_midy},
				global_inputs 	 = {'dt': self.dt},
				outputs 		 = {'out_h': h_new},
				domain 			 = gt.domain.Rectangle((1, 1), (self.m+1, self.n-2)),
				mode 			 = self.backend,
			)
		else:
			stencil_lw = gt.NGStencil(
				definitions_func = self.definitions_lax_wendroff,
				inputs 			 = {'in_h': self.h, 'in_u': self.u, 'in_v': self.v,
									'dx': self.dx, 'dxc': self.dxc,
						  			'dy': self.dy, 'dyc': self.dyc,
								   	'dy1': self.dy1, 'dy1c': self.dy1c,
									'c': self.c, 'c_midy': self.c_midy,
									'f': self.f, 'hs': self.hs, 'tg': self.tg,
									'tg_midx': self.tg_midx, 'tg_midy': self.tg_midy},
				global_inputs 	 = {'dt': self.dt},
				outputs 		 = {'out_h': h_new, 'out_u': u_new, 'out_v': v_new},
				domain 			 = gt.domain.Rectangle((1, 1), (self.m+1, self.n-2)),
				mode 			 = self.backend,
			)
			if self.diffusion:
				stencil_hdiff = gt.NGStencil(
					definitions_func = self.definitions_diffusion,
					inputs 			 = {'in_q': h_ext, 'tmp_q': h_tmp_ext,
										'Ax': self.Ax, 'Ay': self.Ay,
										'Bx': self.Bx, 'By': self.By,
										'Cx': self.Cx, 'Cy': self.Cy},
					global_inputs 	 = {'dt': self.dt},
					outputs 		 = {'out_q': h_new_ext},
					domain 			 = gt.domain.Rectangle((2, 2), (self.m+2, self.n-1)),
					mode 			 = self.backend,
				)
				stencil_udiff = gt.NGStencil(
					definitions_func = self.definitions_diffusion,
					inputs 			 = {'in_q': u_ext, 'tmp_q': u_tmp_ext,
										  'Ax': self.Ax, 'Ay': self.Ay,
										  'Bx': self.Bx, 'By': self.By,
										  'Cx': self.Cx, 'Cy': self.Cy},
					global_inputs 	 = {'dt': self.dt},
					outputs 		 = {'out_q': u_new_ext},
					domain 			 = gt.domain.Rectangle((2, 2), (self.m+2, self.n-1)),
					mode 			 = self.backend,
				)
				stencil_vdiff = gt.NGStencil(
					definitions_func = self.definitions_diffusion,
					inputs 			 = {'in_q': v_ext, 'tmp_q': v_tmp_ext,
										  'Ax': self.Ax, 'Ay': self.Ay,
										  'Bx': self.Bx, 'By': self.By,
										  'Cx': self.Cx, 'Cy': self.Cy},
					global_inputs 	 = {'dt': self.dt},
					outputs 		 = {'out_q': v_new_ext},
					domain 			 = gt.domain.Rectangle((2, 2), (self.m+2, self.n-1)),
					mode 			 = self.backend,
				)

		#
		# Print and save
		#
		if verbose > 0:
			if self.only_advection:
				hmin = self.h[1:-1, :].min()
				hmax = self.h[1:-1, :].max()
				
				print('\n%7.2f (out of %i) hours: min(h) = %12.5f, max(h) = %12.5f'
					  % (0., int(self.t_final / 3600.), hmin, hmax))
			else:
				norm = np.sqrt(self.u[1:-1, :] * self.u[1:-1, :] + 
							   self.v[1:-1, :] * self.v[1:-1, :])
				umax = norm.max()
				
				print('\n%7.2f (out of %i) hours: max(|u|) = %13.8f'
					  % (0., int(self.t_final / 3600.), umax))
								
		# Save
		t_save = np.array([[0]])
		h_save = self.h[1:-1, :, np.newaxis]
		u_save = self.u[1:-1, :, np.newaxis]
		v_save = self.v[1:-1, :, np.newaxis]
								
		#
		# Time marching
		# 
		n, t = 0, 0.
		
		# Save height and stereographic components at the poles
		# This is needed for pole treatment
		# Remark: for the leap-frog scheme, we also need the solution
		# at t = -dt; for the sake of simplicity, we set:
		# h(-dt) = h(0), u(-dt) = u(0), v(-dt) = v(0)
		h_north_now, h_south_now = self.h[0, -1], self.h[0, 0]
		if not self.only_advection:
			u_north_now, u_south_now = self.u_north, self.u_south
			hu_north_now, hu_south_now = h_north_now * u_north_now, h_south_now * u_south_now
			v_north_now, v_south_now = self.v_north, self.v_south
			hv_north_now, hv_south_now = h_north_now * v_north_now, h_south_now * v_south_now

		elapsed_time = 0.

		self.timer.start(name="Time integration", level=2)

		while t < self.t_final and n < self.nmax:
			start = time.time()
			n += 1
					
			# 
			# Compute timestep through CFL condition 
			# Keep track of the old timestep for leap-frog scheme at the poles
			dtold = copy.deepcopy(self.dt)
			
			# Compute flux Jacobian eigenvalues
			eigenx = (np.maximum(np.absolute(self.u[1:-1, :] -
											 np.sqrt(self.g * np.absolute(self.h[1:-1, :]))),
								 np.maximum(np.absolute(self.u[1:-1, :]),
								 			np.absolute(self.u[1:-1, :] +
								 						np.sqrt(self.g *
																np.absolute(self.h[1:-1, :])))))).max()
			eigeny = (np.maximum(np.absolute(self.v[1:-1, :] -
											 np.sqrt(self.g * np.absolute(self.h[1:-1, :]))),
								 np.maximum(np.absolute(self.v[1:-1, :]),
								 			np.absolute(self.v[1:-1, :] +
								 						np.sqrt(self.g *
																np.absolute(self.h[1:-1, :])))))).max()
			
			# Compute timestep					 
			dtmax = np.minimum(self.dx_min/eigenx, self.dy_min/eigeny)
			self.dt.value = self.cfl * dtmax
		
			# If needed, adjust timestep
			if t + self.dt > self.t_final:
				self.dt = gt.Global(self.t_final - t) 
				t = self.t_final
			else:
				t += self.dt.value
				
			# 
			# Update height and stereographic components at the poles 
			# This is needed for pole treatment
			h_north_old, h_south_old = h_north_now, h_south_now
			if not self.only_advection:
				hu_north_old, hu_south_old = hu_north_now, hu_south_now
				hv_north_old, hv_south_old = hv_north_now, hv_south_now
			
			h_north_now, h_south_now = self.h[0, -1], self.h[0, 0]
			if not self.only_advection:
				u_north_now, u_south_now = self.u_north, self.u_south
				hu_north_now, hu_south_now = h_north_now * u_north_now, h_south_now * u_south_now
				v_north_now, v_south_now = self.v_north, self.v_south
				hv_north_now, hv_south_now = h_north_now * v_north_now, h_south_now * v_south_now
			
			# 
			# Update solution at the internal grid points
			#
			if self.only_advection:
				stencil.compute()
			else:
				stencil_lw.compute()
				if self.diffusion:
					self._extend_solution(self.h, h_ext, h_new, h_tmp_ext)
					self._extend_solution(self.u, u_ext, u_new, u_tmp_ext)
					self._extend_solution(self.v, v_ext, v_new, v_tmp_ext)

					stencil_hdiff.compute()
					stencil_udiff.compute()
					stencil_vdiff.compute()

					h_new[1:-1, 1:-1] = h_new_ext[2:-2, 2:-2]
					u_new[1:-1, 1:-1] = u_new_ext[2:-2, 2:-2]
					v_new[1:-1, 1:-1] = v_new_ext[2:-2, 2:-2]

			#
			# North pole treatment
			# 
			dtv = self.dt.value
			dtoldv = dtold.value
			h_north_new = h_north_old + \
						  (dtv + dtoldv) * 2. / (self.dxp * self.m_north * self.m) * \
						  np.sum(self.h[1:-2, -2] * self.v[1:-2, -2])
											
			if not self.only_advection:
				# Compute auxiliary terms
				AU = - 2. / (self.dxp * self.m_north * self.m) * \
					 np.sum(self.h[1:-2, -2] * (self.u[1:-2, -2] * np.sin(self.phi_1d[1:-2]) +
					 						    self.v[1:-2, -2] * np.cos(self.phi_1d[1:-2])) *
					 		self.v[1:-2, -2])
				BU = - self.g / (self.dxp * self.m) * \
					 np.sum((self.h[1:-2, -2] ** 2.) * np.cos(self.phi_1d[1:-2]))
					 
				AV = - 2. / (self.dxp * self.m_north * self.m) * \
					 np.sum(self.h[1:-2, -2] * (- self.u[1:-2, -2] * np.cos(self.phi_1d[1:-2]) +
					 						    self.v[1:-2, -2] * np.sin(self.phi_1d[1:-2])) *
					 		self.v[1:-2, -2])
				BV = - self.g / (self.dxp * self.m) * \
					 np.sum((self.h[1:-2, -2] ** 2.) * np.sin(self.phi_1d[1:-2]))
				fp = self.f[0, -1]
					 
				# Update U
				hu_north_new = 1. / (1. + 0.25 * ((dtoldv + dtv) ** 2.) * (fp ** 2.)) * \
							   ((1. - 0.25 * ((dtoldv + dtv) ** 2.) * (fp ** 2.)) *
								hu_north_old + (dtoldv + dtv) * (AU + BU) +
								(dtoldv + dtv) * fp * hv_north_old +
							    0.5 * ((dtoldv + dtv) ** 2.) * fp * (AV + BV))
				u_north_new = hu_north_new / h_north_new
				self.u_north = u_north_new
				
				# Update V
				hv_north_new = hv_north_old + (dtoldv + dtv) * (AV + BV) - \
							   0.5 * (dtoldv + dtv) * fp * (hu_north_old + hu_north_new)
				v_north_new = hv_north_new / h_north_new
				self.v_north = v_north_new
				
			# 
			# South pole treatment
			# 
			h_south_new = h_south_old - \
						  (dtv + dtoldv) * 2. / (self.dxp * self.m_south * self.m) * \
						  np.sum(self.h[1:-2, 1] * self.v[1:-2, 1])
									
			# Update stereographic components
			if not self.only_advection:
				# Compute auxiliary terms
				AU = - 2. / (self.dxp * self.m_south * self.m) * \
					 np.sum(self.h[1:-2, 1] * (- self.u[1:-2, 1] * np.sin(self.phi_1d[1:-2]) +
					 						   self.v[1:-2, 1] * np.cos(self.phi_1d[1:-2])) *
					 		self.v[1:-2, 1])
				BU = - self.g / (self.dxp * self.m) * \
					 np.sum((self.h[1:-2, 1] ** 2.) * np.cos(self.phi_1d[1:-2]))
					 
				AV = - 2. / (self.dxp * self.m_south * self.m) * \
					 np.sum(self.h[1:-2, 1] * (self.u[1:-2, 1] * np.cos(self.phi_1d[1:-2]) +
					 						  self.v[1:-2, 1] * np.sin(self.phi_1d[1:-2])) *
					 		self.v[1:-2, 1])
				BV = - self.g / (self.dxp * self.m) * \
					 np.sum((self.h[1:-2, 1] ** 2.) * np.sin(self.phi_1d[1:-2]))
				fp = self.f[0, 0]
					 
				# Update U
				hu_south_new = 1. / (1. + 0.25 * ((dtoldv + dtv) ** 2.) * (fp ** 2.)) * \
							 ((1. - 0.25 * ((dtoldv + dtv) ** 2.) * (fp ** 2.)) *
							  hu_south_old + (dtoldv + dtv) * (AU + BU) -
							  (dtoldv + dtv) * fp * hv_south_old -
							  0.5 * ((dtoldv + dtv) ** 2.) * fp * (AV + BV))
				u_south_new = hu_south_new / h_south_new
				self.u_south = u_south_new
				
				# Update V
				hv_south_new = hv_south_old + (dtoldv + dtv) * (AV + BV) + \
							   0.5 * (dtoldv + dtv) * fp * (hu_south_old + hu_south_new)
				v_south_new = hv_south_new / h_south_new
				self.v_south = v_south_new
				
			# 
			# Apply boundary conditions
			#
			self.h[:, 1:-1] = np.concatenate((h_new[-3:-2, 1:-1], 
											  h_new[1:-1, 1:-1], 
											  h_new[2:3, 1:-1]), axis=0)
			self.h[:, -1], self.h[:, 0] = h_north_new, h_south_new	
		
			if not self.only_advection:
				self.u[:, 1:-1] = np.concatenate((u_new[-3:-2, 1:-1], 
												  u_new[1:-1, 1:-1], 
												  u_new[2:3, 1:-1]), axis=0)
				self.u[:, -1] = - u_north_new * np.sin(self.phi_1d) \
								+ v_north_new * np.cos(self.phi_1d)	
				self.u[:, 0]  = - u_south_new * np.sin(self.phi_1d) \
								+ v_south_new * np.cos(self.phi_1d)
		
				self.v[:, 1:-1] = np.concatenate((v_new[-3:-2, 1:-1], 
												  v_new[1:-1, 1:-1], 
												  v_new[2:3, 1:-1]), axis=0)
				self.v[:, -1] = - u_north_new * np.cos(self.phi_1d) \
								- v_north_new * np.sin(self.phi_1d)	
				self.v[:, 0]  = + u_south_new * np.cos(self.phi_1d) \
								+ v_south_new * np.sin(self.phi_1d)
																						  	  			
			end = time.time()
			elapsed_time += (end - start)

			self.timer.start(name="Saving fields during time integration", level=3)
			#
			# Print and save
			# 
			if (verbose > 0 and n % verbose == 0) or t == self.t_final:
				if self.only_advection:
					hmin = self.h[1:-1, :].min()
					hmax = self.h[1:-1, :].max()
			
					print('%7.2f (out of %i) hours: min(h) = %12.5f, max(h) = %12.5f'
						  % (t / 3600., int(self.t_final / 3600.), hmin, hmax))
				else:
					norm = np.sqrt(self.u[1:-1, :] * self.u[1:-1, :] + 
							  	   self.v[1:-1, :] * self.v[1:-1, :])
					umax = norm.max()
					
					print('%7.2f (out of %i) hours: max(|u|) = %13.8f'
						  % (t / 3600., int(self.t_final / 3600.), umax))
					
			if (save > 0 and n % save == 0) or  t == self.t_final:
				t_save = np.concatenate((t_save, np.array([[t]])), axis=0)
				h_save = np.concatenate((h_save, self.h[1:-1, :, np.newaxis]), axis=2)
				if not self.only_advection:
					u_save = np.concatenate((u_save, self.u[1:-1, :, np.newaxis]), axis=2)
					v_save = np.concatenate((v_save, self.v[1:-1, :, np.newaxis]), axis=2)

			self.timer.stop(name="Saving fields during time integration")
		# 
		# Resume 
		#

		self.timer.stop(name="Time integration")
		print('\nTotal number of iterations performed: {}'.format(n))
		print('Average time per iteration: {} ms\n'.format(elapsed_time/n * 1000.))

		self.timer.stop(name="Overall SWES time")
		self.timer.list_timings()

		return t_save, self.phi[1:-1, :], self.theta[1:-1, :], h_save, u_save, v_save


if __name__ == '__main__':
	# Suggested values for $\alpha$ for first and second
    # test cases from Williamson's suite:
    # * 0
    # * 0.05
    # * pi/2 - 0.05
    # * pi/2
	ic = (0, math.pi/2)

	# Suggested simulation's length for Williamson's test cases:
    # * IC 0: 12 days
    # * IC 1: 14 days
	t_final = 12

	# Let's go!
	solver = LaxWendroffSWES(planet=0, t_final=t_final, m=180, n=90, ic=ic,
                             cfl=1, diff=True, backend=gt.mode.NUMPY, dtype=np.float64)
	t, phi, theta, h, u, v = solver.solve(verbose=100, save=100)

	# Save data
	filename = 'results/swes_ic{}.npz'.format(ic[0])
	np.savez(filename, t=t, phi=phi, theta=theta, h=h, u=u, v=v)
