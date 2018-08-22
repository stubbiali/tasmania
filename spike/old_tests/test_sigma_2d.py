"""
Test Sigma2d class. Data are retrieved from COSMO documentation, Part I.
"""
import math
import numpy as np

import namelist as nl
from grids.sigma import Sigma2d as Grid

# Define zonal and vertical domain
domain_x, nx = [0., 500.e3], 51
domain_z, nz = [0.1, 1.] , 50

# Specify the interfacial height separating the terrain-following part of
# the domain from the z-system
pf = 220e2
zf = pf / nl.p_sl

# Define terrain-surface height
#hs = 3000. * np.exp(- (np.linspace(domain_x[0], domain_x[1], n_x) - 5.)**2. / (20.*20.))
hs = '3000. * exp(- (x - 40.)*(x - 40.) / (15.*15.))'

# Instantiate a grid object
g = Grid(domain_x, nx, domain_z, nz, #interface_z = zf, 
		 topo_type = 'gaussian', topo_max_height = 1000., topo_width_x = 50.e3)
		 #topo_type = 'user_defined', topo_str = hs)

# Plot
g.plot()

print('Test passed!')
