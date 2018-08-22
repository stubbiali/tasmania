"""
Test GridXYZ.
"""
from grids.grid_xyz import GridXYZ

domain_x, nx = [-5., 5,], 101
domain_y, ny = [-5., 5.], 101
domain_z, nz = [300., 300. + 50.], 20

topo_str = '1. / (x*x + 1.)'

xyz_grid = GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz,
				   #topo_type = 'schaer', topo_width_x = 1., topo_width_y = 2.)
				   topo_type = 'user_defined', topo_str = topo_str)

xyz_grid._topography.plot(xyz_grid.xy_grid)

print('Test passed!')
