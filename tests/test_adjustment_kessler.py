# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
Script to test the class AdjustmentMicrophysicsKessler.
"""
from datetime import datetime, timedelta
import numpy as np

from tasmania.grids.grid_xyz import GridXYZ
from tasmania.parameterizations.adjustment_microphysics import AdjustmentMicrophysics
from tasmania.storages.state_isentropic import StateIsentropic

domain_x, nx = [0.,500.e3], 100
domain_y, ny = [0.,500.e3], 100
domain_z, nz = [0.,500.e3], 100
grid = GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz)

dt = timedelta(seconds = 5.4)

rho = np.random.rand((nx, ny, nz), dtype = float)
p   = np.random.rand((nx, ny, nz+1), dtype = float)
exn = np.random.rand((nx, ny, nz+1), dtype = float)
T   = np.random.rand((nx, ny, nz), dtype = float)
qv  = np.random.rand((nx, ny, nz), dtype = float)
qc  = np.random.rand((nx, ny, nz), dtype = float)
qr  = np.random.rand((nx, ny, nz), dtype = float)
state = StateIsentropic(datetime(year = 1992, month = 2, day = 20), grid,
						air_density                                 = rho,
						air_pressure                                = p,
						exner_function                              = exn,
						air_temperature                             = T,
						mass_fraction_of_water_vapor_in_air         = qv,
						mass_fraction_of_cloud_liquid_water_in_air  = qc,
						mass_fraction_of_orecipitation_water_in_air = qr)

mp = AdjustmentMicrophysics('kessler', grid, False, gt.mode.NUMPY)

state_new, diagnostics = mp(dt, state)

print('Test passed!')
