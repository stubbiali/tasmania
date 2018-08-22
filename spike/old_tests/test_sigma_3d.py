"""
Test Sigma3d class.
"""
import math
import numpy as np

from grids.sigma import Sigma3d as Grid

# Define zonal, meridional and vertical domain
domain_x, nx = [0.,90.], 91
domain_y, ny = [-30., 45.], 76
domain_z, nz = [.1,1.], 20

# Instantiate a grid object
g = Grid(domain_x, nx, domain_y, ny, domain_z, nz,
		 topo_type = 'gaussian', topo_width_x = 10., topo_width_y = 10.)
print(g.reference_pressure)

print('Test passed!')
