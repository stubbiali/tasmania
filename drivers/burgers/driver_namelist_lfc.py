# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import argparse
import gt4py as gt
import numpy as np
import os
from sympl import DataArray
import tasmania as taz
import time

from drivers.burgers import namelist_lfc


# ============================================================
# The namelist
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    metavar="NAMELIST",
    type=str,
    default="namelist_lfc.py",
    help="The namelist file.",
    dest="namelist",
)
args = parser.parse_args()
namelist = args.namelist.replace("/", ".")
namelist = namelist[:-3] if namelist.endswith(".py") else namelist
exec("import {} as namelist".format(namelist))
nl = locals()["namelist"]
taz.feed_module(target=nl, source=namelist_lfc)

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
    topography_type="flat",
    **nl.backend_settings
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
    backend=nl.backend_settings["backend"],
    dtype=nl.backend_settings["dtype"],
    default_origin=nl.backend_settings["default_origin"],
    managed_memory=nl.backend_settings["managed_memory"],
)
state = zsf(nl.init_time, cgrid)

# set the initial state as reference state for the handler of
# the lateral boundary conditions
domain.horizontal_boundary.reference_state = state

# ============================================================
# The slow tendency components
# ============================================================
# component calculating the Laplacian of the velocity
diff = taz.BurgersHorizontalDiffusion(
    domain,
    "numerical",
    nl.diffusion_type,
    nl.diffusion_coeff,
    **nl.backend_settings
)

# ============================================================
# The dynamical core
# ============================================================
dycore = taz.BurgersDynamicalCore(
    domain,
    fast_tendency_component=None,
    time_integration_scheme=nl.time_integration_scheme,
    flux_scheme=nl.flux_scheme,
    **nl.backend_settings
)

# ============================================================
# A NetCDF monitor
# ============================================================
if nl.save and nl.filename is not None:
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

# dict operator
dict_op = taz.DataArrayDictOperator(**nl.backend_settings)

for i in range(nt):
    compute_time_start = time.time()

    tendencies, _ = diff(state)

    # step the solution
    dict_op.copy(state, dycore(state, tendencies, dt))

    state["time"] = nl.init_time + (i + 1) * dt

    compute_time += time.time() - compute_time_start

    if (
        (nl.print_frequency > 0)
        and ((i + 1) % nl.print_frequency == 0)
        or i + 1 == nt
    ):
        dx = pgrid.dx.to_units("m").data.item()
        dy = pgrid.dy.to_units("m").data.item()

        u = state["x_velocity"].to_units("m s^-1").data[3:-3, 3:-3, :]
        v = state["y_velocity"].to_units("m s^-1").data[3:-3, 3:-3, :]

        max_u = u.max()
        max_v = v.max()

        # print useful info
        print(
            "Iteration {:6d}: max(u) = {:12.10E} m/s, max(v) = {:12.10E} m/s".format(
                i + 1, max_u.item(), max_v.item()
            )
        )

    # shortcuts
    to_save = (
        nl.save
        and nl.filename is not None
        and (
            (nl.save_frequency > 0 and (i + 1) % nl.save_frequency == 0)
            or i + 1 == nt
        )
    )

    if to_save:
        # save the solution
        netcdf_monitor.store(state)

print("Simulation successfully completed. HOORAY!")

# ============================================================
# Post-processing
# ============================================================
# dump the solution to file
if nl.save and nl.filename is not None:
    netcdf_monitor.write()

# stop chronometer
wall_time = time.time() - wall_time_start

# restore numpy
gt.storage.restore_numpy()

# compute the error
try:
    u = np.asarray(state["x_velocity"].data)
except ValueError:
    u = state["x_velocity"].data.get()
uex = zsof(state["time"], cgrid, field_name="x_velocity", field_units="m s^-1")
print("RMSE(u) = {:.5E} m/s".format(np.linalg.norm(u - uex) / np.sqrt(u.size)))

# print logs
print("Total wall time: {}.".format(taz.get_time_string(wall_time, False)))
print("Compute time: {}.".format(taz.get_time_string(compute_time, True)))
