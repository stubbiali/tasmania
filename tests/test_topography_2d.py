""" 
Test Topography2d class. 
"""
import numpy as np

from grids.grid_xy import GridXY
from grids.topography import Topography2d

domain_x, nx = [0.,10.], 1e2+1
domain_y, ny = [0.,10.], 1e2+1

grid = GridXY(domain_x, nx, domain_y, ny)

topo_str = '3000. * exp(- (x-3.)*(x-3.) - (y-5.)(y-5.))'

hs = Topography2d(grid, 
				  #topo_type = 'schaer', topo_width_x = 1., topo_width_y = 2.)
				  topo_type = 'user_defined', topo_str = topo_str)

hs.plot(grid)

print('Test passed!')
