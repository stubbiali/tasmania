"""
Test SLEVE2d class. Data are retrieved from COSMO documentation, Part I.
"""
import math
import numpy as np

import namelist as nl
from grids.sleve import SLEVE2d as Grid

# Define zonal and vertical domain
domain_x, nx = [0., 100.], 101
domain_z, nz = [15000., 0.] , 20

# Specify the interfacial height separating the terrain-following part of
# the domain from the z-system
zf = 11360. 

# Define terrain-surface height
topo_str = '3000. * exp(- (x - 50.)*(x - 50.) / (20.*20.) )'

# Instantiate a grid object
g = Grid(domain_x, nx, domain_z, nz, interface_z = zf, 
		 #topo_type = 'gaussian', topo_max_height = 3000., topo_width_x = 20)
		topo_type = 'user_defined', topo_str = topo_str)

# Plot
g.plot()

print('Test passed!')
