import os
import numpy as np
import gridtools as gt
import matplotlib.pyplot as plt


class BurgersSolver:
	"""
	GridTools4Py-v3 implementation of a finite difference solver for the following 
	initial and boundary value problem (IBVP) for Burgers' equations: 
	\begin{align*}
		& \dfrac{\partial \boldsymbol{u}}{\partial t} + \left( \boldsymbol{u} \cdot \nabla \right) \boldsymbol{u} = \boldsymbol{u} \hspace*{0.5cm} \text{for $\boldsymbol{u} = \boldsymbol{u}(\boldsymbol{x}, \, t) \in \mathbb{R}^2$, $\boldsymbol{x} \in \Omega = (0, \, 2) \times (0, \, 2)$, $t \in \times (0, \, T) \rightarrow \mathbb{R}^2$} \, , \\
		& \boldsymbol{u}(\boldsymbol{x}, \, 0) =
		\begin{cases}
			(4, \, 1)^T \hspace*{0.5cm} \text{if $\boldsymbol{x} \in [0.5, \, 1] \times [0.5, \, 1]$} \\
			(1, \, 2)^T \hspace*{0.5cm} \text{otherwise}
		\end{cases} \, , \\
		& \boldsymbol{u}(\boldsymbol{x}, \, t) = \boldsymbol{0} \hspace*{0.5cm} \text{for $\boldsymbol{x} \in \partial \Omega$, $t \in [0, \, T]$} \, .
	\end{align*}
	"""

	def __init__(self, nx, ny, tf, nt, backend = gt.mode.DEBUG):
		"""
		Constructor setting physical and numerical parameters.

		:param nx		Grid points along the first dimension
		:param ny		Grid points along the second dimension
		:param tf		Simulation final time
		:param nt		Timesteps
		:param backend	GridTools4Py backend; options are:
						gt.mode.ALPHA = GridTools4Py-v2
						gt.mode.DEBUG = Python debug
						gt.mode.NUMPY = Numpy (vectorized)
		"""
		# Set the domain
		self._south_west = (0,0)
		self._north_east = (2,2)
		
		#
		# Build spatial grid
		#
		assert ((nx > 1) and (ny > 1)), \
			"Number of grid points along each direction must be greater than one."
		
		# Discretize x-axis
		self._nx = nx		
		self._dx = gt.Global((self._north_east[0] - self._south_west[0])/(self._nx - 1))
		self._x = np.linspace(self._south_west[0], self._north_east[0], self._nx)

		# Discretize y-axis
		self._ny = ny
		self._dy = gt.Global((self._north_east[1]- self._south_west[1])/(self._ny - 1))
		self._y = np.linspace(self._south_west[1], self._north_east[1], self._ny)
		
		# Build grid
		self._xv, self._yv = np.meshgrid(self._x, self._y, indexing = 'ij')	
		
		#
		# Discretize time interval
		# 	
		assert (tf > 0), "Final time must be strictly positive."
		assert (nt > 1), "Number of timesteps must be greater than one."
		
		self._tf = float(tf)
		self._nt = nt
		self._dt = gt.Global(self._tf/(self._nt - 1))
		self._t = np.linspace(0., self._tf, self._nt)

		#
		# Initialize solution
		#
		self.u = np.zeros((self._nx, self._ny), float)
		self.v = np.zeros((self._nx, self._ny), float)
		self._apply_initial_conditions()
		self._apply_boundary_conditions()

		#
		# Set backend
		#
		self._backend = backend
				

	def _apply_initial_conditions(self):
		"""
		Set initial conditions on velocity.
		"""
		for i in range(self._nx):
			for j in range(self._ny):
				if ((0.5 <= self._x[i]) and (self._x[i] <= 1.0) and \
					(0.5 <= self._y[j]) and (self._y[j] <= 1.0)):
					self.u[i,j] = 4.0
					self.v[i,j] = 1.0
				else:
					self.u[i,j] = 1.0
					self.v[i,j] = 2.0


	def _apply_boundary_conditions(self):
		"""
		Apply (Dirichlet) boundary conditions. 
		"""
		self.u[0,:] = self.u[-1,:] = self.u[:,0] = self.u[:,-1] = 0.
		self.v[0,:] = self.v[-1,:] = self.v[:,0] = self.v[:,-1] = 0.

	
	def set_backend(self, backend):
		"""
		Set GridTools4Py backend.

		:param backend	GridTools4Py backend; options are:
						gt.mode.ALPHA = GridTools4Py-v2
						gt.mode.DEBUG = Python debug
						gt.mode.NUMPY = Numpy (vectorized)
		"""
		self._backend = backend


	def definitions_stencil(self, dt, dx, dy, in_u, in_v):
		"""
		Stencil. The method uses backward finite-difference in space and forward finite-difference in time.

		:param in_u	Input x-velocity
		:param in_v	Input y-velocity
		
		:return Update x-velocity
		:return Update y-velocity
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()

		# Outputs
		out_u = gt.Equation()
		out_v = gt.Equation()

		# Stages
		out_u[i,j] = in_u[i,j] - dt * in_u[i,j] * (in_u[i,j] - in_u[i-1,j]) / dx \
						       - dt * in_v[i,j] * (in_u[i,j] - in_u[i,j-1]) / dy			   
		out_v[i,j] = in_v[i,j] - dt * in_u[i,j] * (in_v[i,j] - in_v[i-1,j]) / dx \
						   	   - dt * in_v[i,j] * (in_v[i,j] - in_v[i,j-1]) / dy

		return out_u, out_v


	def solve(self, be_verbose = 5, do_plot = 0):
		"""
		Forward the solution.

		:param be_verbose		if positive, print information about the solution every be_verbose timesteps
		:param do_plot			if positive, plot the solution every do_plot timesteps
		
		:return x- and y-velocity at final timestep
		"""

		# 
		# Print and plot
		#
		if be_verbose > 0:
			norm = np.sqrt(self.u*self.u + self.v*self.v)
			umax = norm.max()
			print("Time %6.3f s: max(|u|) = %16.8f" \
					% (self._t[0], umax))
		
		if do_plot > 0:
			plt.figure()
			Q = plt.quiver(self._yv, self._xv, np.transpose(self.u), np.transpose(self.v))
			plt.xlabel('x [m]')
			plt.ylabel('y [m]')
			plt.title('Velocity field [m/s], time 0.0 [s]')
			plt.axis([self._south_west[0], self._north_east[0], self._south_west[1], self._north_east[1]])
			plt.draw()

		#
		# Allocate output arrays and initialize the stencil
		#
		unew = np.zeros((self._nx, self._ny), float)
		vnew = np.zeros((self._nx, self._ny), float)
		stencil = gt.NGStencil(definitions_func = self.definitions_stencil,
		   					   inputs = {"in_u": self.u, "in_v": self.v},
							   global_inputs = {"dt": self._dt, "dx": self._dx, "dy": self._dy},
		   					   outputs = {"out_u": unew, "out_v": vnew},
		   					   domain = gt.domain.Rectangle((1, 1), (self._nx-1, self._ny-1)),
		   					   mode = self._backend)
			
		#
		# Time marching
		#
		for ts in range(1, self._nt):		
			# Run the stencil
			stencil.compute()

			# Update velocity
			self.u[:,:] = unew[:,:]
			self.v[:,:] = vnew[:,:]

			# Apply boundary conditions
			self._apply_boundary_conditions()

			# 
			# Print and plot
			# 
			if be_verbose > 0 and ts % be_verbose == 0:
				norm = np.sqrt(self.u*self.u + self.v*self.v);
				umax = norm.max();
				print("Time %6.3f s: max(|u|) = %16.8f" \
						% (self._t[ts], umax))
			
			if do_plot > 0 and ((ts % do_plot == 0) or (ts == self._nt-1)):
				Q.set_UVC(np.transpose(self.u), np.transpose(self.v));
				plt.title('Velocity field [m/s], time %3.3f [s]' % self._t[ts]);
				plt.axis([self._south_west[0], self._north_east[0], self._south_west[1], self._north_east[1]])
				plt.draw();
				plt.pause(0.1);
				
		return self.u, self.v
		

#
# Test script
#
bg = BurgersSolver(161, 161, 0.5, 201, gt.mode.NUMPY)
bg.solve()
