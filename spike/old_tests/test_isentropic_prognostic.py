# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
Test IsentropicPrognostic class.
"""
import numpy as np

from dycore.isentropic_prognostic import IsentropicPrognostic
from python.grids import XYZGrid as Grid
import gridtools as gt
from namelist import datatype

#
# Settings
#
domain_x, nx = [0.0, 10.0], 100
domain_y, ny = [-5.0, 5.0], 100
domain_z, nz = [0.0, 20.0], 20
moist = True
horizontal_bcs = "relaxed"
scheme = "maccormack"

#
# Initialization
#
grid = Grid(domain_x, nx, domain_y, ny, domain_z, nz)

dt = gt.Global(2.0)

if scheme in ["upwind", "leapfrog"]:
    nb = 1
elif scheme in ["maccormack"]:
    nb = 2

if horizontal_bcs in ["periodic"]:
    ni = nx + 2 * nb
    nj = ny + 2 * nb
    nk = nz
elif horizontal_bcs in ["relaxed"]:
    ni = nx
    nj = ny
    nk = nz

s = np.ones((ni, nj, nk), dtype=datatype)
u = np.zeros((ni + 1, nj, nk), dtype=datatype)
v = np.zeros((ni, nj + 1, nk), dtype=datatype)
mtg = np.zeros((ni, nj, nk), dtype=datatype)
U = np.zeros((ni, nj, nk), dtype=datatype)
V = np.zeros((ni, nj, nk), dtype=datatype)
Qv = np.zeros((ni, nj, nk), dtype=datatype)
Qc = np.zeros((ni, nj, nk), dtype=datatype)
Qr = np.zeros((ni, nj, nk), dtype=datatype)

#
# Test
#
prog = IsentropicPrognostic(
    grid, moist=moist, horizontal_bcs=horizontal_bcs, scheme=scheme
)
if moist:
    out_s, out_U, out_V, out_Qv, out_Qc, out_Qr = prog.step_forward(
        dt,
        s,
        u,
        v,
        mtg,
        U,
        V,
        Qv,
        Qc,
        Qr,
        old_s=s,
        old_U=U,
        old_V=V,
        old_Qv=Qv,
        old_Qc=Qc,
        old_Qr=Qr,
    )
else:
    out_s, out_U, out_V = prog.step_forward(
        dt, s, u, v, mtg, U, V, old_s=s, old_U=U, old_V=V
    )

print(out_s[22, 41, 1])
