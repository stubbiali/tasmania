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
import click
import numpy as np
import os

from sympl._core.data_array import DataArray

import gt4py as gt

import tasmania as taz

from drivers.burgers import namelist_fc
from drivers.benchmarking.utils import inject_backend


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
        DataArray([0, 1], dims="z", attrs={"units": "1"}),
        1,
        horizontal_boundary_type=nl.hb_type,
        nb=nl.nb,
        horizontal_boundary_kwargs=nl.hb_kwargs,
        topography_type="flat",
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )
    pgrid = domain.physical_grid
    ngrid = domain.numerical_grid

    # ============================================================
    # The initial state
    # ============================================================
    zsof = taz.ZhaoSolutionFactory(nl.init_time, nl.diffusion_coeff)
    zsf = taz.ZhaoStateFactory(
        nl.init_time,
        nl.diffusion_coeff,
        backend=nl.backend,
        storage_options=nl.so,
    )
    state = zsf(nl.init_time, ngrid)

    # set the initial state as reference state for the handler of
    # the lateral boundary conditions
    domain.horizontal_boundary.reference_state = state

    # ============================================================
    # The fast tendencies
    # ============================================================
    # component calculating the Laplacian of the velocity
    diff = taz.BurgersHorizontalDiffusion(
        domain,
        "numerical",
        nl.diffusion_type,
        nl.diffusion_coeff,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    # ============================================================
    # The dynamical core
    # ============================================================
    dycore = taz.BurgersDynamicalCore(
        domain,
        fast_tendency_component=diff,
        time_integration_scheme=nl.time_integration_scheme,
        flux_scheme=nl.flux_scheme,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
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

    # first time iterate
    state_new = dycore(state, {}, dt)
    state_new["time"] = nl.init_time + dt

    for i in range(1, nt):
        # swap old and new states
        state, state_new = state_new, state

        # step the solution
        dycore(state, {}, dt, out_state=state_new)
        state_new["time"] = nl.init_time + (i + 1) * dt

        if (
            (nl.print_frequency > 0)
            and ((i + 1) % nl.print_frequency == 0)
            or i + 1 == nt
        ):
            dx = pgrid.dx.to_units("m").data.item()
            dy = pgrid.dy.to_units("m").data.item()

            u = state_new["x_velocity"].to_units("m s^-1").data[3:-3, 3:-3, :]
            v = state_new["y_velocity"].to_units("m s^-1").data[3:-3, 3:-3, :]

            max_u = u.max()
            max_v = v.max()

            # print useful info
            print(
                f"Iteration {i+1:6d}: max(u) = {max_u.item():12.10E} m/s, "
                f"max(v) = {max_v.item():12.10E} m/s"
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
            netcdf_monitor.store(state_new)

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # dump the solution to file
    if nl.save and nl.filename is not None:
        netcdf_monitor.write()

    # compute the error
    u = taz.to_numpy(state_new["x_velocity"].data)
    uex = zsof(
        state_new["time"], ngrid, field_name="x_velocity", field_units="m s^-1"
    )
    gt.storage.restore_numpy()
    print(
        "RMSE(u) = {:.5E} m/s".format(
            np.linalg.norm(u - uex) / np.sqrt(u.size)
        )
    )


if __name__ == "__main__":
    main()
