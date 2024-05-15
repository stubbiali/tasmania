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
from datetime import datetime
import numpy as np
from sympl import DataArray

from tasmania.python.domain.domain import Domain
from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils.storage import get_dataarray_3d
from tasmania.python.utils.time import Timer


def main(nx, ny, nz, niter, backend):
    bo = BackendOptions(rebuild=False, verbose=True)
    so = StorageOptions(dtype=float)

    domain_x = DataArray([0, 100], dims="x", attrs={"units": "m"})
    domain_y = DataArray([0, 100], dims="y", attrs={"units": "m"})
    domain_z = DataArray([0, 100], dims="z", attrs={"units": "m"})
    storage_shape = (nx, ny, nz)
    domain = Domain(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        horizontal_boundary_type="identity",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    grid = domain.numerical_grid

    u_np = np.random.rand(*storage_shape)
    u = as_storage(backend, data=u_np, storage_options=so)
    u_da = get_dataarray_3d(u, grid, "m s^-1", set_coordinates=False)
    v_np = np.random.rand(*storage_shape)
    v = as_storage(backend, data=v_np, storage_options=so)
    v_da = get_dataarray_3d(v, grid, "m s^-1", set_coordinates=False)
    state = {
        "time": datetime(year=1992, month=2, day=20),
        "x_velocity": u_da,
        "y_velocity": v_da,
    }

    tnd_u = get_dataarray_3d(
        zeros(backend, shape=storage_shape, storage_options=so),
        grid,
        "m s^-2",
        set_coordinates=False,
    )
    tnd_v = get_dataarray_3d(
        zeros(backend, shape=storage_shape, storage_options=so),
        grid,
        "m s^-2",
        set_coordinates=False,
    )
    out_tendencies = {"x_velocity": tnd_u, "y_velocity": tnd_v}

    smag = Smagorinsky2d(
        domain,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    _, _ = smag(state)

    Timer.start(label="main")
    for _ in range(niter):
        _, _ = smag(state, out_tendencies=out_tendencies)
    Timer.stop(label="main")

    print(f"Wall-clock time: {Timer.get_time('main', 's')} s")


if __name__ == "__main__":
    main(256, 256, 256, 100, "gt4py:gtx86")
