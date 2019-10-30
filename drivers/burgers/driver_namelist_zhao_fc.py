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
import gridtools as gt
import numpy as np
import os
from sympl import DataArray
import tasmania as taz
import time

try:
    from . import namelist_zhao_fc as nl
except (ImportError, ModuleNotFoundError):
    import namelist_zhao_fc as nl


gt.storage.prepare_numpy()

# ============================================================
# The underlying domain
# ============================================================
domain = taz.Domain(
    nl.domain_x,
    nl.nx,
    nl.domain_y,
    nl.ny,
    DataArray([0, 1], dims="z", attrs={"units": "1"}),
    1,
    horizontal_boundary_type=nl.hb_type,
    nb=nl.nb,
    horizontal_boundary_kwargs=nl.hb_kwargs,
    topography_type="flat_terrain",
    backend=nl.gt_kwargs["backend"],
    dtype=nl.gt_kwargs["dtype"],
)
pgrid = domain.physical_grid
cgrid = domain.numerical_grid

# ============================================================
# The initial state
# ============================================================
zsof = taz.ZhaoSolutionFactory(nl.init_time, nl.diffusion_coeff)
zsf = taz.ZhaoStateFactory(
    nl.init_time,
    nl.diffusion_coeff,
    backend=nl.gt_kwargs["backend"],
    dtype=nl.gt_kwargs["dtype"],
    halo=nl.gt_kwargs["halo"],
)
state = zsf(nl.init_time, cgrid)

# set the initial state as reference state for the handler of
# the lateral boundary conditions
domain.horizontal_boundary.reference_state = state

# ============================================================
# The intermediate tendencies
# ============================================================
# component calculating the Laplacian of the velocity
diff = taz.BurgersHorizontalDiffusion(
    domain, "numerical", nl.diffusion_type, nl.diffusion_coeff, **nl.gt_kwargs
)

# ============================================================
# The dynamical core
# ============================================================
dycore = taz.BurgersDynamicalCore(
    domain,
    intermediate_tendencies=diff,
    time_integration_scheme=nl.time_integration_scheme,
    flux_scheme=nl.flux_scheme,
    **nl.gt_kwargs
)

# ============================================================
# A NetCDF monitor
# ============================================================
if nl.filename is not None and nl.save_frequency > 0:
    if os.path.exists(nl.filename):
        os.remove(nl.filename)

    netcdf_monitor = taz.NetCDFMonitor(nl.filename, domain, "physical")
    netcdf_monitor.store(state)

# ============================================================
# Time-marching
# ============================================================
dt = nl.timestep
nt = nl.niter

# timer
wall_time_start = time.time()
compute_time = 0.0

for i in range(nt):
    compute_time_start = time.time()

    # step the solution
    taz.dict_copy(state, dycore(state, {}, dt))

    state["time"] = nl.init_time + (i + 1) * dt

    compute_time += time.time() - compute_time_start

    if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
        dx = pgrid.dx.to_units("m").values.item()
        dy = pgrid.dy.to_units("m").values.item()

        u = state["x_velocity"].to_units("m s^-1").values.data[3:-3, 3:-3, :]
        v = state["y_velocity"].to_units("m s^-1").values.data[3:-3, 3:-3, :]

        # uex = zsof(state["time"], cgrid, field_name="x_velocity")[3:-3, 3:-3, :]
        # vex = zsof(state["time"], cgrid, field_name="y_velocity")[3:-3, 3:-3, :]
        # state_ex = zsf(state["time"], cgrid)
        # uex = state_ex["x_velocity"].to_units("m s^-1").values[3:-3, 3:-3, :]
        # vex = state_ex["y_velocity"].to_units("m s^-1").values[3:-3, 3:-3, :]

        # err_u = np.linalg.norm(u - uex) * np.sqrt(dx * dy)
        # err_v = np.linalg.norm(v - vex) * np.sqrt(dx * dy)

        err_u = u.max()
        err_v = v.max()

        # print useful info
        print(
            "Iteration {:6d}: ||u - uex|| = {:12.10E} m/s, ||v - vex|| = {:12.10E} m/s".format(
                i + 1, err_u.item(), err_v.item()
            )
        )

        # umax, vmax = u[3:-3, 3:-3].max(), v[3:-3, 3:-3].max()
        # umin, vmin = u[3:-3, 3:-3].min(), v[3:-3, 3:-3].min()
        # print(
        #     "Iteration {:6d}: umax = {:12.10E}, umin = {:12.10E}, vmax = {:12.10E}, vmin = {:12.10E}".format(
        #         i + 1, umax, umin, vmax, vmin
        #     )
        # )

    # shortcuts
    to_save = (nl.filename is not None) and (
        ((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt
    )

    if to_save:
        # save the solution
        netcdf_monitor.store(state)

print("Simulation successfully completed. HOORAY!")

# ============================================================
# Post-processing
# ============================================================
# dump the solution to file
if nl.filename is not None and nl.save_frequency > 0:
    netcdf_monitor.write()

# stop the timer
wall_time = time.time() - wall_time_start

# print logs
print("Total wall time: {}.".format(taz.get_time_string(wall_time)))
print("Compute time: {}.".format(taz.get_time_string(compute_time)))
