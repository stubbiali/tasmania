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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains a small example to experiment with mpi one sided
"""
import numpy as np
from mpi4py import MPI

nb = 2
sx = 6
sy = 6
nn = 4
nindex = [None] * nn

buffers = {}
buffers["test"] = [np.zeros((sx, nb, 1)),
                  np.zeros((sx, nb, 1)),
                  np.zeros((nb, sy, 1)),
                  np.zeros((nb, sy, 1))]
windows = {}
windows["test"] = [MPI.Win.Create(buffers["test"][0], comm=MPI.COMM_WORLD),
                   MPI.Win.Create(buffers["test"][1], comm=MPI.COMM_WORLD),
                   MPI.Win.Create(buffers["test"][2], comm=MPI.COMM_WORLD),
                   MPI.Win.Create(buffers["test"][3], comm=MPI.COMM_WORLD)]


temp_buffer = [np.zeros_like(buffers["test"][0]),
               np.zeros_like(buffers["test"][1]),
               np.zeros_like(buffers["test"][2]),
               np.zeros_like(buffers["test"][3])]


if MPI.COMM_WORLD.Get_rank() == 0:
    test_arr = np.arange((sx+nb+nb) * (sy+nb+nb)).reshape((sx+nb+nb, sy+nb+nb, 1))
    nindex = [None, 1, None, 2]

    buffers["test"][0][:] = test_arr[nb:-nb, -nb-nb:-nb, :]
    buffers["test"][1][:] = test_arr[nb:-nb, nb:nb+nb, :]
    buffers["test"][2][:] = test_arr[-nb-nb:-nb, nb:-nb, :]
    buffers["test"][3][:] = test_arr[nb:nb+nb, nb:-nb, :]

    for d in range(nn):
        windows["test"][d].Fence()
        if nindex[d] is not None:
            windows["test"][d].Get(origin=temp_buffer[d], target_rank=nindex[d])
        windows["test"][d].Fence()

    test_arr[nb:-nb, 0:nb] = temp_buffer[0][:]
    test_arr[nb:-nb, -nb:] = temp_buffer[1][:]
    test_arr[0:nb, nb:-nb] = temp_buffer[2][:]
    test_arr[-nb:, nb:-nb] = temp_buffer[3][:]


if MPI.COMM_WORLD.Get_rank() == 1:
    test_arr = 200 + np.arange((sx+nb+nb) * (sy+nb+nb)).reshape((sx+nb+nb, sy+nb+nb, 1))

    nindex = [0, None, None, None]

    buffers["test"][0][:] = test_arr[nb:-nb, -nb-nb:-nb, :]
    buffers["test"][1][:] = test_arr[nb:-nb, nb:nb+nb, :]
    buffers["test"][2][:] = test_arr[-nb-nb:-nb, nb:-nb, :]
    buffers["test"][3][:] = test_arr[nb:nb+nb, nb:-nb, :]

    for d in range(nn):
        windows["test"][d].Fence()
        if nindex[d] is not None:
            windows["test"][d].Get(origin=temp_buffer[d], target_rank=nindex[d])
        windows["test"][d].Fence()

    test_arr[nb:-nb, 0:nb] = temp_buffer[0][:]
    test_arr[nb:-nb, -nb:] = temp_buffer[1][:]
    test_arr[0:nb, nb:-nb] = temp_buffer[2][:]
    test_arr[-nb:, nb:-nb] = temp_buffer[3][:]

if MPI.COMM_WORLD.Get_rank() == 2:
    test_arr = 400 + np.arange((sx+nb+nb) * (sy+nb+nb)).reshape((sx+nb+nb, sy+nb+nb, 1))
    nindex = [None, None, 0, None]

    buffers["test"][0][:] = test_arr[nb:-nb, -nb-nb:-nb, :]
    buffers["test"][1][:] = test_arr[nb:-nb, nb:nb+nb, :]
    buffers["test"][2][:] = test_arr[-nb-nb:-nb, nb:-nb, :]
    buffers["test"][3][:] = test_arr[nb:nb+nb, nb:-nb, :]

    for d in range(nn):
        windows["test"][d].Fence()
        if nindex[d] is not None:
            windows["test"][d].Get(origin=temp_buffer[d], target_rank=nindex[d])
        windows["test"][d].Fence()

    test_arr[nb:-nb, 0:nb] = temp_buffer[0][:]
    test_arr[nb:-nb, -nb:] = temp_buffer[1][:]
    test_arr[0:nb, nb:-nb] = temp_buffer[2][:]
    test_arr[-nb:, nb:-nb] = temp_buffer[3][:]

print(MPI.COMM_WORLD.Get_rank(), "\n", test_arr.transpose())



