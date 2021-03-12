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
import gt4py as gt
import numpy as np
import tasmania.python.utils.time
from sympl import DataArray
import tasmania as taz

from drivers.benchmarking.burgers import namelist_ps
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
    default="namelist_ps.py",
    help="The namelist file.",
)
@click.option("--no-log", is_flag=True, help="Disable log.")
def main(backend=None, namelist="namelist_ps.py", no_log=False):
    # ============================================================
    # The namelist
    # ============================================================
    _namelist = namelist.replace("/", ".")
    _namelist = _namelist[:-3] if _namelist.endswith(".py") else _namelist
    exec(f"import {_namelist} as namelist_module")
    nl = locals()["namelist_module"]
    taz.feed_module(target=nl, source=namelist_ps)
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
        intermediate_tendency_component=None,
        time_integration_scheme=nl.time_integration_scheme,
        flux_scheme=nl.flux_scheme,
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
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    # wrap the component in a ParallelSplitting object
    physics = taz.ParallelSplitting(
        taz.TimeIntegrationOptions(
            component=diff,
            scheme=nl.physics_time_integration_scheme,
            enforce_horizontal_boundary=True,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        ),
        execution_policy="serial",
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
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

    for i in range(nt):
        # start timing
        tasmania.python.utils.time.Timer.start("compute_time")

        # calculate the dynamics
        tasmania.python.utils.time.Timer.start("dynamics")
        state_prv = dycore(state, {}, dt)
        tasmania.python.utils.time.Timer.stop("dynamics")

        # calculate the physics
        tasmania.python.utils.time.Timer.start("physics")
        physics(state, state_prv, dt)
        tasmania.python.utils.time.Timer.stop("physics")

        # update the state
        tasmania.python.utils.time.Timer.start("state_update")
        dict_op.copy(state, state_prv)
        state["time"] = nl.init_time + (i + 1) * dt
        tasmania.python.utils.time.Timer.stop("state_update")

        # stop timing
        tasmania.python.utils.time.Timer.stop("compute_time")

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # compute the error
    u = taz.to_numpy(state["x_velocity"].data)
    uex = zsof(
        state["time"], ngrid, field_name="x_velocity", field_units="m s^-1"
    )
    print(f"RMSE(u) = {np.linalg.norm(u - uex) / np.sqrt(u.size):.5E} m/s")

    # print logs
    print(
        f"Compute time: "
        f"{tasmania.python.utils.time.Timer.get_time('compute_time', 's'):.3f}"
        f" s."
    )

    if not no_log:
        # save to file
        exec_info_to_csv(nl.exec_info_csv, nl.backend, nl.bo)
        run_info_to_csv(
            nl.run_info_csv,
            backend,
            tasmania.python.utils.time.Timer.get_time("compute_time", "s"),
        )
        tasmania.python.utils.time.Timer.log(nl.log_txt, "s")


if __name__ == "__main__":
    main()