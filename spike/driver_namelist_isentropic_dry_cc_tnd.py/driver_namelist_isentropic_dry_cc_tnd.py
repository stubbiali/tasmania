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
import numpy as np
import os
import tasmania as taz
import time

import namelist_isentropic_dry_cc as nl

# ============================================================
# The underlying grid
# ============================================================
grid = taz.GridXYZ(
    nl.domain_x,
    nl.nx,
    nl.domain_y,
    nl.ny,
    nl.domain_z,
    nl.nz,
    topo_type=nl.topo_type,
    topo_time=nl.topo_time,
    topo_kwargs=nl.topo_kwargs,
    dtype=nl.dtype,
)

# ============================================================
# The initial state
# ============================================================
if nl.isothermal:
    state = taz.get_isothermal_isentropic_state(
        grid,
        nl.init_time,
        nl.init_x_velocity,
        nl.init_y_velocity,
        nl.init_temperature,
        moist=False,
        precipitation=False,
        dtype=nl.dtype,
    )
else:
    state = taz.get_default_isentropic_state(
        grid,
        nl.init_time,
        nl.init_x_velocity,
        nl.init_y_velocity,
        nl.init_brunt_vaisala,
        moist=False,
        precipitation=False,
        dtype=nl.dtype,
    )

# ============================================================
# The intermediate tendencies
# ============================================================
args = []

# Component calculating the pressure gradient in isentropic_prognostic coordinates
order = 4 if nl.horizontal_flux_scheme == "fifth_order_upwind" else 2
pg = taz.ConservativeIsentropicPressureGradient(
    grid,
    order=order,
    horizontal_boundary_type=nl.horizontal_boundary_type,
    backend=nl.backend,
    dtype=nl.dtype,
)
args.append(pg)

if nl.coriolis:
    # Component calculating the Coriolis acceleration
    cf = taz.ConservativeIsentropicCoriolis(
        grid, coriolis_parameter=nl.coriolis_parameter, dtype=nl.dtype
    )
    args.append(cf)

# Wrap the components in a ConcurrentCoupling object
inter_tends = taz.ConcurrentCoupling(*args, execution_policy="serial")


# ============================================================
# The intermediate diagnostics
# ============================================================
args = []

# Component retrieving the diagnostic variables
pt = state["air_pressure_on_interface_levels"][0, 0, 0]
dv = taz.IsentropicDiagnostics(
    grid, moist=nl.moist, pt=pt, backend=nl.backend, dtype=nl.dtype
)
args.append(dv)

# Wrap the components in a DiagnosticComponentComposite object
inter_diags = taz.DiagnosticComponentComposite(*args)

# ============================================================
# The dynamical core
# ============================================================
dycore = taz.HomogeneousIsentropicDynamicalCore(
    grid,
    time_units="s",
    moist=nl.moist,
    # parameterizations
    intermediate_tendencies=inter_tends,
    intermediate_diagnostics=inter_diags,
    substeps=nl.substeps,
    fast_tendencies=None,
    fast_diagnostics=None,
    # numerical scheme
    time_integration_scheme=nl.time_integration_scheme,
    horizontal_flux_scheme=nl.horizontal_flux_scheme,
    horizontal_boundary_type=nl.horizontal_boundary_type,
    # vertical damping
    damp=nl.damp,
    damp_type=nl.damp_type,
    damp_depth=nl.damp_depth,
    damp_max=nl.damp_max,
    damp_at_every_stage=nl.damp_at_every_stage,
    # horizontal smoothing
    smooth=nl.smooth,
    smooth_type=nl.smooth_type,
    smooth_damp_depth=nl.smooth_damp_depth,
    smooth_coeff=nl.smooth_coeff,
    smooth_coeff_max=nl.smooth_coeff_max,
    smooth_at_every_stage=nl.smooth_at_every_stage,
    # backend settings
    backend=nl.backend,
    dtype=nl.dtype,
)

# ============================================================
# A NetCDF monitor
# ============================================================
if nl.filename is not None and nl.save_frequency > 0:
    if os.path.exists(nl.filename):
        os.remove(nl.filename)

    netcdf_monitor = taz.NetCDFMonitor(
        nl.filename, grid, store_names=nl.store_names
    )
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

    # Update the (time-dependent) topography
    dycore.update_topography((i + 1) * dt)

    # Step the solution
    state_new = dycore(state, {}, dt)

    # Retrieve the physical tendencies
    tnd_pg, _ = pg(state)
    if nl.coriolis:
        tnd_cf, _ = cf(state)
        tnd_phys = add(tnd_pg, tnd_cf, unshared_variables_in_output=True)
    else:
        tnd_phys = tnd_pg

    # Retrieve the dynamical tendencies
    state_update = subtract(
        state_new, state, unshared_variables_in_output=True
    )
    tnd_total = multiply(1 / dt.total_seconds(), state_update)
    tnd_total["x_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    tnd_total["y_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    tnd_dyn = subtract(tnd_total, tnd_phys, unshared_variables_in_output=True)

    # Update the state
    taz.dict_update(state, state_new)

    compute_time += time.time() - compute_time_start

    if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
        # Print useful info
        print(
            "Iteration {:6d}: tnd_pg_su = {:8.4f}, tnd_cf_su = {:8.4f}, "
            "tnd_dyn_su = {:8.4f}".format(
                i + 1,
                np.linalg.norm(
                    tnd_pg["x_momentum_isentropic"][20:61, 20:61, :]
                ),
                np.linalg.norm(
                    tnd_cf["x_momentum_isentropic"][20:61, 20:61, :]
                ),
                np.linalg.norm(
                    tnd_dyn["x_momentum_isentropic"][20:61, 20:61, :]
                ),
            )
        )
        print(
            "Iteration {:6d}: tnd_pg_sv = {:8.4f}, tnd_cf_sv = {:8.4f}, "
            "tnd_dyn_sv = {:8.4f}".format(
                i + 1,
                np.linalg.norm(
                    tnd_pg["y_momentum_isentropic"][20:61, 20:61, :]
                ),
                np.linalg.norm(
                    tnd_cf["y_momentum_isentropic"][20:61, 20:61, :]
                ),
                np.linalg.norm(
                    tnd_dyn["y_momentum_isentropic"][20:61, 20:61, :]
                ),
            )
        )

    # Shortcuts
    to_save = (nl.filename is not None) and (
        ((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0))
        or i + 1 == nt
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
