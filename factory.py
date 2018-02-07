"""
A multi-purpose factory.
"""
import grids.gal_chen
import grids.sigma
import grids.sleve
import grids.xyz_grid

import dycore.horizontal_boundary
import dycore.isentropic_dycore
import dycore.isentropic_flux
import dycore.vertical_damping

import gridtools as gt

class Factory:
	"""
	Multi-purpose factory class. It provides static methods returning an instance
	of the required grid, vertical damper, dynamical core, and more.
	"""
	@staticmethod
	def instantiate_dynamical_core(self, model, grid, imoist, horizontal_boundary_type, scheme, backend,
								   idamp, damp_type, damp_depth, damp_max, 
								   idiff, diff_coeff, diff_coeff_moist, diff_max):
		"""
		Based on the complete model name specified by :data:`model`, instantiate the dynamical core.
		"""
		if model == 'isentropic':
			return dycore.isentropic_dycore.IsentropicDynamicalCore(grid, imoist, horizontal_boundary_type, scheme, backend,
								   									idamp, damp_type, damp_depth, damp_max, 
								   									idiff, diff_coeff, diff_coeff_moist, diff_max)
		else:
			raise ValueError('Unknown model. Available options: ''isentropic''.')

	@staticmethod
	def instantiate_flux(self, model, scheme, grid, imoist):
		"""
		Based on the complete model name specified by :data:`model` and the numerical scheme specified by :data:`scheme`,
		instantiate the class in charge of computing the numerical flux.
		"""
		if model == 'isentropic':
			if scheme == 'upwind':
				return dycore.isentropic_flux.Upwind(grid, imoist)
			elif scheme == 'leapfrog':
				return dycore.isentropic_flux.Leapfrog(grid, imoist)
			else:
				return dycore.isentropic_flux.MacCormack(grid, imoist)
		else:
			raise ValueError('Unknown model. Available options: ''isentropic''.')
			
	@staticmethod
	def instantiate_grid(self, model, domain_x, nx, domain_y, ny, domain_z, nz, 
				 		 z_interface, topo_type, topo_time, topo_max_height,
						 topo_width_x, topo_width_y, topo_str):
		"""
		Based on the complete model name specified by :data:`model`, instantiate the underlying grid.
		"""
		if model == 'isentropic':
			return grids.xyz_grid.XYZGrid(domain_x, nx, domain_y, ny, domain_z, nz,
										  units_x = 'm', dims_x = 'x',
										  units_y = 'm', dims_y = 'y',
										  units_z = 'K', dims_z = 'potential_temperature',
										  z_interface = z_interface,
										  topo_type = topo_type, topo_time = topo_time, topo_max_height = topo_max_height, 
										  topo_width_x = topo_width_x, topo_width_y = topo_width_y, topo_str = topo_str)
		else:
			raise ValueError('Unknown model. Available options: ''isentropic''.')

	@staticmethod
	def instantiate_horizontal_boundary(self, horizontal_boundary_type, grid, nb):
		"""
		Based on the boundary conditions type specified by :data:`horizontal_boundary_type`, 
		return an instance of the class taking care of the horizontal boundary conditions.
		"""
		if horizontal_boundary_type == 'periodic':
			return dycore.horizontal_boundary.Periodic(grid, nb)
		elif horizontal_boundary_type == 'relaxed':
			return dycore.horizontal_boundary.Relaxed(grid, nb)
		else:
			raise ValueError('Unknown boundary conditions type. Available options: ''periodic'', ''relaxed''.')

	@staticmethod
	def instantiate_parametrization(self, param_name):
		pass

	@staticmethod
	def instantiate_vertical_damper(self, damp_type, grid, nb, damp_depth, damp_max, backend):
		"""
		Based on the damping scheme specified by :data:`damp_type`, return an instance of the class implementing
		vertical damping.
		"""
		if damp_type == 'rayleigh':
			return dycore.vertical_damping.Rayleigh(grid, nb, damp_depth, damp_max, backend)
		else:
			raise ValueError('Unknown damping scheme. Available options: ''rayleigh''.')
