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
Script for testing the classes StateIsentropicNonconservative and StateIsentropicConservative.
"""
from datetime import datetime
import numpy as np

from grids.grid_xyz import GridXYZ
from storages.grid_data import GridData
from storages.state_isentropic_nonconservative import StateIsentropicNonconservative
from storages.state_isentropic_conservative import StateIsentropicConservative

# Instantiate the grid
domain_x, nx = [0,1], 20
domain_y, ny = [0,1], 20
domain_z, nz = [0,1], 10
g = GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz)

# Allocate model variables
s   = np.zeros((nx  , ny  , nz  ), dtype = float)
u   = np.zeros((nx+1, ny  , nz  ), dtype = float)
u_  = .5 * (u[:-1,:,:] + u[1:,:,:])
v   = np.zeros((nx  , ny+1, nz  ), dtype = float)
v_  = .5 * (v[:,:-1,:] + v[:,1:,:])
p   = np.zeros((nx  , ny  , nz+1), dtype = float)
exn = np.zeros((nx  , ny  , nz+1), dtype = float)
mtg = np.zeros((nx  , ny  , nz  ), dtype = float)
h   = np.zeros((nx  , ny  , nz+1), dtype = float)
U   = np.zeros((nx  , ny  , nz  ), dtype = float)
V   = np.zeros((nx  , ny  , nz  ), dtype = float)
qv  = np.zeros((nx  , ny  , nz  ), dtype = float)
qc  = np.zeros((nx  , ny  , nz  ), dtype = float)
qr  = np.zeros((nx  , ny  , nz  ), dtype = float)
Qv  = np.zeros((nx  , ny  , nz  ), dtype = float)
Qc  = np.zeros((nx  , ny  , nz  ), dtype = float)
Qr  = np.zeros((nx  , ny  , nz  ), dtype = float)

# Instantiate nonconservative model state
sinc = StateIsentropicNonconservative(datetime(year = 1992, month = 2, day = 20), g, s, u, u_, v, v_, p, exn, mtg, h, qv, qc, qr)

# Instantiate conservative model state
sic = StateIsentropicConservative(datetime(year = 1992, month = 2, day = 20), g, s, u, U, v, V, p, exn, mtg, h, Qv, Qc, Qr)

# Update the states
upd = GridData(datetime(year = 1992, month = 2, day = 20), g, x_velocity_unstaggered = u_, height = None)
sinc.update(upd)
sic.update(upd)

print('Test passed!')
