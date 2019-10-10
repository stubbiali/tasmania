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
import numpy as np
import os
from sympl import DataArray
import tasmania as taz
import time

try:
    from . import namelist_zhao_ps as nl
except (ImportError, ModuleNotFoundError):
    import namelist_zhao_ps as nl

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
# The dynamical core
# ============================================================
dycore = taz.BurgersDynamicalCore(
    domain,
    intermediate_tendencies=None,
    time_integration_scheme=nl.time_integration_scheme,
    flux_scheme=nl.flux_scheme,
    **nl.gt_kwargs
)

# ============================================================
# The physics
# ============================================================
# component calculating the Laplacian of the velocity
diff = taz.BurgersHorizontalDiffusion(
<<<<<<< HEAD
	domain, 'numerical', nl.diffusion_type, nl.diffusion_coeff,
	nl.backend, nl.dtype
=======
    domain, "numerical", nl.diffusion_type, nl.diffusion_coeff, **nl.gt_kwargs
>>>>>>> gt4py_framework
)

# Wrap the component in a ParallelSplitting object
physics = taz.ParallelSplitting(
    {
        "component": diff,
        "time_integrator": nl.physics_time_integration_scheme,
        "enforce_horizontal_boundary": True,
        "substeps": 1,
        "backend": nl.gt_kwargs["backend"],
        "halo": nl.gt_kwargs["halo"],
    }
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

wall_time_start = time.time()
compute_time = 0.0

for i in range(nt):
    compute_time_start = time.time()

    # Calculate the dynamics
    state_prv = dycore(state, {}, dt)

    # Calculate the physics
    physics(state, state_prv, dt)

    compute_time += time.time() - compute_time_start

    if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0) or i == nt - 1:
        dx = pgrid.dx.to_units("m").values.item()
        dy = pgrid.dy.to_units("m").values.item()

        u = state["x_velocity"].to_units("m s^-1").values[3:-3, 3:-3, :]
        v = state["y_velocity"].to_units("m s^-1").values[3:-3, 3:-3, :]

        uex = zsof(state["time"], cgrid, field_name="x_velocity")[3:-3, 3:-3, :]
        vex = zsof(state["time"], cgrid, field_name="y_velocity")[3:-3, 3:-3, :]

        err_u = np.linalg.norm(u - uex) * np.sqrt(dx * dy)
        err_v = np.linalg.norm(v - vex) * np.sqrt(dx * dy)

        # Print useful info
        print(
            "Iteration {:6d}: ||u - uex|| = {:8.4E} m/s, ||v - vex|| = {:8.4E} m/s".format(
                i + 1, err_u, err_v
            )
        )

    # Shortcuts
    to_save = (nl.filename is not None) and (
        ((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt
    )

    if to_save:
        # Save the solution
        netcdf_monitor.store(state)

print("Simulation successfully completed. HOORAY!")

# ============================================================
# Post-processing
# ============================================================
# Dump the solution to file
if nl.filename is not None and nl.save_frequency > 0:
    netcdf_monitor.write()

# Stop chronometer
wall_time = time.time() - wall_time_start

# Print logs
print("Total wall time: {}.".format(taz.get_time_string(wall_time)))
print("Compute time: {}.".format(taz.get_time_string(compute_time)))
