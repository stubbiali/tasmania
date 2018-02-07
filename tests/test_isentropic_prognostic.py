"""
Test IsentropicPrognostic class.
"""
import numpy as np

from dycore.isentropic_prognostic import IsentropicPrognostic
from grids.xyz_grid import XYZGrid as Grid
import gridtools as gt
from namelist import datatype

# 
# Settings
#
domain_x, nx = [0.,10.], 100
domain_y, ny = [-5.,5.], 100
domain_z, nz = [0.,20.], 20
moist = True
horizontal_bcs = 'relaxed'
scheme = 'maccormack'

#
# Initialization
#
grid = Grid(domain_x, nx, domain_y, ny, domain_z, nz)

dt = gt.Global(2.)

if scheme in ['upwind', 'leapfrog']:
	nb = 1
elif scheme in ['maccormack']:
	nb = 2

if horizontal_bcs in ['periodic']:
	ni = nx + 2*nb
	nj = ny + 2*nb
	nk = nz
elif horizontal_bcs in ['relaxed']:
	ni = nx
	nj = ny
	nk = nz

s   = np.ones((ni, nj, nk), dtype = datatype)
u   = np.zeros((ni+1, nj, nk), dtype = datatype)
v   = np.zeros((ni, nj+1, nk), dtype = datatype)
mtg = np.zeros((ni, nj, nk), dtype = datatype)
U   = np.zeros((ni, nj, nk), dtype = datatype)
V   = np.zeros((ni, nj, nk), dtype = datatype)
Qv  = np.zeros((ni, nj, nk), dtype = datatype)
Qc  = np.zeros((ni, nj, nk), dtype = datatype)
Qr  = np.zeros((ni, nj, nk), dtype = datatype)

#
# Test 
#
prog = IsentropicPrognostic(grid, moist = moist, horizontal_bcs = horizontal_bcs, scheme = scheme)
if moist:
	out_s, out_U, out_V, out_Qv, out_Qc, out_Qr = \
		prog.step_forward(dt, s, u, v, mtg, U, V, Qv, Qc, Qr, \
		 		 	 	  old_s = s, old_U = U, old_V = V, old_Qv = Qv, old_Qc = Qc, old_Qr = Qr)
else:
	out_s, out_U, out_V = prog.step_forward(dt, s, u, v, mtg, U, V, old_s = s, old_U = U, old_V = V)
	
print(out_s[22,41,1])
