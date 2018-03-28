"""
Script to test the class AdjustmentMicrophysicsKessler.
"""
from datetime import datetime, timedelta
import numpy as np

import gridtools as gt
from tasmania.grids.grid_xyz import GridXYZ
from tasmania.parameterizations.adjustment_microphysics import AdjustmentMicrophysics
from tasmania.storages.state_isentropic import StateIsentropic

domain_x, nx = [0.,500.e3], 100
domain_y, ny = [0.,500.e3], 100
domain_z, nz = [0.,500.e3], 100
grid = GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz)

dt = timedelta(seconds = 5.4)

rho = np.random.rand(nx, ny, nz)
p   = np.random.rand(nx, ny, nz+1)
exn = np.random.rand(nx, ny, nz+1)
T   = 273. + np.random.rand(nx, ny, nz)
qv  = np.random.rand(nx, ny, nz)
qc  = np.random.rand(nx, ny, nz)
qr  = np.random.rand(nx, ny, nz)
state = StateIsentropic(datetime(year = 1992, month = 2, day = 20), grid,
						air_density                                 = rho,
						air_pressure                                = p,
						exner_function                              = exn,
						air_temperature                             = T,
						mass_fraction_of_water_vapor_in_air         = qv,
						mass_fraction_of_cloud_liquid_water_in_air  = qc,
						mass_fraction_of_precipitation_water_in_air = qr)

mp = AdjustmentMicrophysics.factory('kessler', grid, False, gt.mode.NUMPY)
mp.time_levels = 1

state_new, diagnostics = mp(dt, state)

print('Test passed!')
