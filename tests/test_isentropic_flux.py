"""
Test IsentropicFlux class.
"""
from dycore.isentropic_flux import IsentropicFlux as Flux
from grids.xyz_grid import XYZGrid as Grid
import gridtools as gt

# 
# Grid instantiation
#
domain_x, nx = [0.,10.], 101
domain_y, ny = [-5.,5.], 51
domain_z, nz = [0.,20.], 21
grid = Grid(domain_x, nx, domain_y, ny, domain_z, nz)

#
# GT4Py initialization
#
i = gt.Index()
j = gt.Index()
k = gt.Index()

dt = gt.Global(1.)

s   = gt.Equation()
u   = gt.Equation()
v   = gt.Equation()
mtg = gt.Equation()
U   = gt.Equation()
V   = gt.Equation()
Qv  = gt.Equation()
Qc  = gt.Equation()
Qr  = gt.Equation()

#
# Test 
#
flux = Flux(grid, moist = True, flux_type = 'maccormack')

F1_x, F2_x, F3_x, F4_x, F5_x, F6_x, F1_y, F2_y, F3_y, F4_y, F5_y, F6_y = \
	flux(i, j, k, dt, s, u, v, mtg, U, V, Qv, Qc, Qr)
print('F1_x: ' + F1_x.get_name())
print('F2_x: ' + F2_x.get_name())
print('F3_x: ' + F3_x.get_name())
print('F4_x: ' + F4_x.get_name())
print('F5_x: ' + F5_x.get_name())
print('F6_x: ' + F6_x.get_name() + '\n')
print('F1_y: ' + F1_y.get_name())
print('F2_y: ' + F2_y.get_name())
print('F3_y: ' + F3_y.get_name())
print('F4_y: ' + F4_y.get_name())
print('F5_y: ' + F5_y.get_name())
print('F6_y: ' + F6_y.get_name() + '\n')
