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
import click
from datetime import timedelta
import numpy as np

from sympl._core.data_array import DataArray
from sympl._core.time import Timer

import tasmania as taz

from drivers.benchmarking.burgers import namelist_sus
from drivers.benchmarking.utils import (
    exec_info_to_csv,
    inject_backend,
    run_info_to_csv,
)


@click.command()
@click.option("-b", "--backend", type=str, default=None, help="The backend.")
@click.option(
    "-n",
    "--namelist",
    type=str,
    default="namelist_sus.py",
    help="The namelist file.",
)
@click.option("--no-log", is_flag=True, help="Disable log.")
def main(backend=None, namelist="namelist_sus.py", no_log=False):
    # ============================================================
    # The namelist
    # ============================================================
    _namelist = namelist.replace("/", ".")
    _namelist = _namelist[:-3] if _namelist.endswith(".py") else _namelist
    exec(f"import {_namelist} as namelist_module")
    nl = locals()["namelist_module"]
    taz.feed_module(target=nl, source=namelist_sus)
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
    # The dynamical core
    # ============================================================
    dycore = taz.BurgersDynamicalCore(
        domain,
        fast_tendency_component=None,
        time_integration_scheme=nl.time_integration_scheme,
        flux_scheme=nl.flux_scheme,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    # ============================================================
    # The physics
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

    # Wrap the component in a SequentialUpdateSplitting object
    physics = taz.SequentialUpdateSplitting(
        taz.TimeIntegrationOptions(
            component=diff,
            scheme=nl.physics_time_integration_scheme,
            enforce_horizontal_boundary=True,
            substeps=1,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

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
    state_new = dycore(state, {}, timedelta(seconds=0.0))
    physics(state_new, timedelta(seconds=0.0))

    # reset timers
    Timer.reset()

    for i in range(nt):
        # start timing
        Timer.start("compute_time")

        # swap old and new states
        state, state_new = state_new, state

        # calculate the dynamics
        dycore(state, {}, dt, out_state=state_new)
        state_new["time"] = nl.init_time + i * dt

        # calculate the physics
        physics(state_new, dt)
        state_new["time"] = nl.init_time + (i + 1) * dt

        # stop timing
        Timer.stop()

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # compute the error
    u = taz.to_numpy(state_new["x_velocity"].data)
    # uex = zsof(
    #     state["time"], ngrid, field_name="x_velocity", field_units="m s^-1"
    # )
    # print(f"RMSE(u) = {np.linalg.norm(u - uex) / np.sqrt(u.size):.5E} m/s")
    print(f"Validation: max(u) = {u.max():.8f} m/s")

    # print logs
    print(f"Compute time: {Timer.get_time('compute_time', 's'):.3f} s.")
    print(f"Stencil time: {Timer.get_time('stencil', 's'):.3f} s.")

    if not no_log:
        # save to file
        exec_info_to_csv(nl.exec_info_csv, nl.backend, nl.bo)
        run_info_to_csv(
            nl.run_info_csv, backend, Timer.get_time("compute_time", "s")
        )
        run_info_to_csv(
            nl.stencil_info_csv, backend, Timer.get_time("stencil", "s")
        )
        Timer.log(nl.log_txt, "s")


if __name__ == "__main__":
    main()
