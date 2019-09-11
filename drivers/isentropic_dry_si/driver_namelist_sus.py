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
import os
import tasmania as taz
import time

import namelist_sus as nl

# ============================================================
# The underlying domain
# ============================================================
domain = taz.Domain(
    nl.domain_x,
    nl.nx,
    nl.domain_y,
    nl.ny,
    nl.domain_z,
    nl.nz,
    horizontal_boundary_type=nl.hb_type,
    nb=nl.nb,
    horizontal_boundary_kwargs=nl.hb_kwargs,
    topography_type=nl.topo_type,
    topography_kwargs=nl.topo_kwargs,
    dtype=nl.dtype,
)
pgrid = domain.physical_grid
cgrid = domain.numerical_grid

# ============================================================
# The initial state
# ============================================================
state = taz.get_isentropic_state_from_brunt_vaisala_frequency(
    cgrid,
    nl.init_time,
    nl.x_velocity,
    nl.y_velocity,
    nl.brunt_vaisala,
    moist=False,
    precipitation=False,
    dtype=nl.dtype,
)
domain.horizontal_boundary.reference_state = state

# ============================================================
# The physics
# ============================================================
args = []
ptis = nl.physics_time_integration_scheme

# component retrieving the diagnostic variables
pt = state["air_pressure_on_interface_levels"][0, 0, 0]
dv = taz.IsentropicDiagnostics(
    domain,
    grid_type="numerical",
    moist=False,
    pt=pt,
    backend=nl.backend,
    dtype=nl.dtype,
)
args.append({"component": dv})

if nl.coriolis:
    # component calculating the Coriolis acceleration
    cf = taz.IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        coriolis_parameter=nl.coriolis_parameter,
        backend=nl.backend,
        dtype=nl.dtype,
    )
    args.append({"component": cf, "time_integrator": ptis, "substeps": 1})

if nl.smooth:
    # component performing the horizontal smoothing
    hs = taz.IsentropicHorizontalSmoothing(
        domain,
        nl.smooth_type,
        nl.smooth_coeff,
        nl.smooth_coeff_max,
        nl.smooth_damp_depth,
        moist=False,
        backend=nl.backend,
        dtype=nl.dtype,
    )
    args.append({"component": hs})

if nl.diff:
    # component calculating tendencies due to numerical diffusion
    hd = taz.IsentropicHorizontalDiffusion(
        domain,
        nl.diff_type,
        nl.diff_coeff,
        nl.diff_coeff_max,
        nl.diff_damp_depth,
        moist=False,
        backend=nl.backend,
        dtype=nl.dtype,
    )
    args.append({"component": hd, "time_integrator": ptis, "substeps": 1})

if nl.turbulence:
    # component implementing the Smagorinsky turbulence model
    turb = taz.IsentropicSmagorinsky(
        domain,
        "numerical",
        smagorinsky_constant=nl.smagorinsky_constant,
        backend=nl.backend,
        dtype=nl.dtype,
    )
    args.append({"component": turb, "time_integrator": ptis, "substeps": 1})

if nl.coriolis or nl.smooth or nl.diff or nl.turbulence:
    # component retrieving the velocity components
    ivc = taz.IsentropicVelocityComponents(domain, backend=nl.backend, dtype=nl.dtype)
    args.append({"component": ivc})

# wrap the components in a SequentialUpdateSplitting object
physics = taz.SequentialUpdateSplitting(*args)

# ============================================================
# The dynamical core
# ============================================================
dycore = taz.IsentropicDynamicalCore(
    domain,
    moist=False,
    # numerical scheme
    time_integration_scheme=nl.time_integration_scheme,
    horizontal_flux_scheme=nl.horizontal_flux_scheme,
    time_integration_properties={
        "pt": state["air_pressure_on_interface_levels"][0, 0, 0],
        "eps": nl.eps,
    },
    # vertical damping
    damp=nl.damp,
    damp_type=nl.damp_type,
    damp_depth=nl.damp_depth,
    damp_max=nl.damp_max,
    damp_at_every_stage=nl.damp_at_every_stage,
    # horizontal smoothing
    smooth=False,
    # backend settings
    backend=nl.backend,
    dtype=nl.dtype,
)

# ============================================================
# A NetCDF monitor
# ============================================================
if nl.filename is not None:
    if os.path.exists(nl.filename):
        os.remove(nl.filename)

    netcdf_monitor = taz.NetCDFMonitor(
        nl.filename, domain, "physical", store_names=nl.store_names
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

    # update the (time-dependent) topography
    dycore.update_topography((i + 1) * dt)

    # compute the dynamics
    state_new = dycore(state, {}, dt)

    # ensure the state is still defined at the current time level
    state_new["time"] = nl.init_time + i * dt

    # compute the physics
    physics(state_new, dt)

    # ensure the state is defined at the next time level
    assert state_new["time"] == state["time"] + dt

    # update the state
    taz.dict_update(state, state_new)

    compute_time += time.time() - compute_time_start

    if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
        u = (
            state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[...]
            / state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[...]
        )
        v = (
            state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[...]
            / state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[...]
        )

        umax, umin = u.max(), u.min()
        vmax, vmin = v.max(), v.min()
        cfl = max(
            umax * dt.total_seconds() / pgrid.dx.to_units("m").values.item(),
            vmax * dt.total_seconds() / pgrid.dy.to_units("m").values.item(),
        )

        # print useful info
        print(
            "Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, "
            "vmax = {:8.4f} m/s, vmin = {:8.4f} m/s".format(
                i + 1, cfl, umax, umin, vmax, vmin
            )
        )

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
if nl.filename is not None:
    netcdf_monitor.write()

# stop chronometer
wall_time = time.time() - wall_time_start

# print logs
print("Total wall time: {}.".format(taz.get_time_string(wall_time)))
print("Compute time: {}.".format(taz.get_time_string(compute_time)))
