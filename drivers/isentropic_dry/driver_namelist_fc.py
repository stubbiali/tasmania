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
import click
from datetime import timedelta
import os

from sympl._core.data_array import DataArray
from sympl._core.time import Timer

import tasmania as taz

from drivers.benchmarking.utils import inject_backend
from drivers.isentropic_dry import namelist_fc
from drivers.isentropic_dry.utils import print_info


@click.command()
@click.option("-b", "--backend", type=str, default=None, help="The backend.")
@click.option(
    "-n",
    "--namelist",
    type=str,
    default="namelist_fc.py",
    help="The namelist file.",
)
@click.option("--no-log", is_flag=True, help="Disable log.")
def main(backend=None, namelist="namelist_fc.py", no_log=False):
    # ============================================================
    # The namelist
    # ============================================================
    _namelist = namelist.replace("/", ".")
    _namelist = _namelist[:-3] if _namelist.endswith(".py") else _namelist
    exec(f"import {_namelist} as namelist_module")
    nl = locals()["namelist_module"]
    taz.feed_module(target=nl, source=namelist_fc)
    inject_backend(nl, backend)

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
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )
    pgrid = domain.physical_grid
    ngrid = domain.numerical_grid
    storage_shape = (ngrid.nx + 1, ngrid.ny + 1, ngrid.nz + 1)

    # ============================================================
    # The initial state
    # ============================================================
    state = taz.get_isentropic_state_from_brunt_vaisala_frequency(
        ngrid,
        nl.init_time,
        nl.x_velocity,
        nl.y_velocity,
        nl.brunt_vaisala,
        moist=False,
        backend=nl.backend,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    domain.horizontal_boundary.reference_state = state

    # ============================================================
    # The fast tendencies
    # ============================================================
    args = []

    if nl.coriolis:
        # component calculating the Coriolis acceleration
        cf = taz.IsentropicConservativeCoriolis(
            domain,
            grid_type="numerical",
            coriolis_parameter=nl.coriolis_parameter,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(cf)

    if nl.diff:
        # component calculating tendencies due to numerical diffusion
        diff = taz.IsentropicHorizontalDiffusion(
            domain,
            nl.diff_type,
            nl.diff_coeff,
            nl.diff_coeff_max,
            nl.diff_damp_depth,
            moist=False,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(diff)

    if nl.turbulence:
        # component implementing the Smagorinsky turbulence model
        turb = taz.IsentropicSmagorinsky(
            domain,
            nl.smagorinsky_constant,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(turb)

    # wrap the components in a ConcurrentCoupling object
    fast_tends = taz.ConcurrentCoupling(
        *args,
        execution_policy="serial",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so
    )

    # ============================================================
    # The fast diagnostics
    # ============================================================
    # component retrieving the diagnostic variables
    pt = DataArray(
        state["air_pressure_on_interface_levels"].data[0, 0, 0],
        attrs={"units": "Pa"},
    )
    dv = taz.IsentropicDiagnostics(
        domain,
        grid_type="numerical",
        moist=False,
        pt=pt,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )

    # ============================================================
    # The slow diagnostics
    # ============================================================
    args = []

    if nl.smooth:
        # component performing the horizontal smoothing
        hs = taz.IsentropicHorizontalSmoothing(
            domain,
            nl.smooth_type,
            nl.smooth_coeff,
            nl.smooth_coeff_max,
            nl.smooth_damp_depth,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(hs)

        # component calculating the velocity components
        vc = taz.IsentropicVelocityComponents(
            domain,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(vc)

    if len(args) > 0:
        # wrap the components in a ConcurrentCoupling object
        slow_diags = taz.ConcurrentCoupling(*args, execution_policy="serial")
    else:
        slow_diags = None

    # ============================================================
    # The dynamical core
    # ============================================================
    dycore = taz.IsentropicDynamicalCore(
        domain,
        moist=False,
        # parameterizations
        fast_tendency_component=fast_tends,
        fast_diagnostic_component=dv,
        substeps=nl.substeps,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        # numerical scheme
        time_integration_scheme=nl.time_integration_scheme,
        horizontal_flux_scheme=nl.horizontal_flux_scheme,
        time_integration_properties={
            "pt": pt,
            "eps": nl.eps,
            "a": nl.a,
            "b": nl.b,
            "c": nl.c,
        },
        # vertical damping
        damp=nl.damp,
        damp_type=nl.damp_type,
        damp_depth=nl.damp_depth,
        damp_max=nl.damp_max,
        damp_at_every_stage=nl.damp_at_every_stage,
        # backend settings
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )

    # ============================================================
    # NetCDF monitor
    # ============================================================
    if nl.save and nl.filename is not None:
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

    # dict operator
    dict_op = taz.DataArrayDictOperator(
        backend=nl.backend, backend_options=nl.bo, storage_options=nl.so
    )

    # warm up caches
    dycore.update_topography(timedelta(seconds=0.0))
    state_new = dycore(state, {}, timedelta(seconds=0.0))
    if slow_diags is not None:
        _, diagnostics = slow_diags(state_new, timedelta(seconds=0.0))
        dict_op.update_swap(state_new, diagnostics)

    # reset timers
    Timer.reset()

    for i in range(nt):
        # start timing
        Timer.start(label="compute_time")

        # swap the old and new state
        state, state_new = state_new, state

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # calculate the dynamics
        dycore(state, {}, dt, out_state=state_new)

        # calculate the slow physics
        if slow_diags is not None:
            slow_diags(state_new, dt, out_diagnostics=diagnostics)
            dict_op.update_swap(state_new, diagnostics)

        # stop timing
        Timer.stop(label="compute_time")

        # print useful info
        print_info(dt, i, nl, pgrid, state_new)

        # shortcuts
        to_save = (
            nl.save
            and (nl.filename is not None)
            and (
                (
                    (nl.save_frequency > 0)
                    and ((i + 1) % nl.save_frequency == 0)
                )
                # i + 1 in nl.save_iterations
                # or i + 1 == nt
            )
        )

        if to_save:
            # save the solution
            netcdf_monitor.store(state_new)

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # dump the solution to file
    if nl.save and nl.filename is not None:
        netcdf_monitor.write()

    # print logs
    print(f"Compute time: {Timer.get_time('compute_time', 's'):.3f} s.")
    if nl.logfile is not None:
        taz.Timer.log(logfile=nl.logfile, units="s")


if __name__ == "__main__":
    main()
