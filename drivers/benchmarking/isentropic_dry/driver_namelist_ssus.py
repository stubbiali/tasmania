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

from drivers.benchmarking.isentropic_dry import namelist_ssus
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
    default="namelist_ssus.py",
    help="The namelist file.",
)
@click.option("--no-log", is_flag=True, help="Disable log.")
def main(backend=None, namelist="namelist_ssus.py", no_log=False):
    # ============================================================
    # The namelist
    # ============================================================
    _namelist = namelist.replace("/", ".")
    _namelist = _namelist[:-3] if _namelist.endswith(".py") else _namelist
    exec(f"import {_namelist} as namelist_module")
    nl = locals()["namelist_module"]
    taz.feed_module(target=nl, source=namelist_ssus)
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
    # The dynamics
    # ============================================================
    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
    dycore = taz.IsentropicDynamicalCore(
        domain,
        moist=False,
        # parameterizations
        fast_tendency_component=None,
        fast_diagnostic_component=None,
        substeps=nl.substeps,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        # numerical scheme
        time_integration_scheme=nl.time_integration_scheme,
        horizontal_flux_scheme=nl.horizontal_flux_scheme,
        time_integration_properties={"pt": pt, "eps": nl.eps},
        # vertical damping
        damp=nl.damp,
        damp_type=nl.damp_type,
        damp_depth=nl.damp_depth,
        damp_max=nl.damp_max,
        damp_at_every_stage=nl.damp_at_every_stage,
        # horizontal smoothing
        smooth=False,
        smooth_moist=False,
        # backend settings
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )

    # ============================================================
    # The physics
    # ============================================================
    args_before_dynamics = []
    args_after_dynamics = []
    ptis = nl.physics_time_integration_scheme

    # component retrieving the diagnostic variables
    dv = taz.IsentropicDiagnostics(
        domain,
        grid_type="numerical",
        moist=False,
        pt=pt,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args_after_dynamics.append(taz.TimeIntegrationOptions(component=dv))

    # component calculating the Coriolis acceleration
    cf = taz.IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        coriolis_parameter=nl.coriolis_parameter,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args_before_dynamics.append(
        taz.TimeIntegrationOptions(
            component=cf,
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )
    args_after_dynamics.append(
        taz.TimeIntegrationOptions(
            component=cf,
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    # component performing the horizontal smoothing
    hs = taz.IsentropicHorizontalSmoothing(
        domain,
        nl.smooth_type,
        nl.smooth_coeff,
        nl.smooth_coeff_max,
        nl.smooth_damp_depth,
        moist=False,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args_after_dynamics.append(taz.TimeIntegrationOptions(component=hs))

    # component implementing the Smagorinsky turbulence model
    turb = taz.IsentropicSmagorinsky(
        domain,
        nl.smagorinsky_constant,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args_before_dynamics.append(
        taz.TimeIntegrationOptions(
            component=turb,
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )
    args_after_dynamics.append(
        taz.TimeIntegrationOptions(
            component=turb,
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    iargs_before_dynamics = args_before_dynamics[::-1]

    # component retrieving the velocity components
    ivc = taz.IsentropicVelocityComponents(
        domain,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    iargs_before_dynamics.append(taz.TimeIntegrationOptions(component=ivc))

    # wrap the components in two SequentialUpdateSplitting objects
    physics_before_dynamics = taz.SequentialUpdateSplitting(
        *iargs_before_dynamics
    )
    physics_after_dynamics = taz.SequentialUpdateSplitting(
        *args_after_dynamics
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
        tasmania.python.utils.time.Timer.start(label="compute_time")

        # auxiliary state
        state_aux = {}
        state_aux.update(state)

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # compute the physics before the dynamics
        tasmania.python.utils.time.Timer.start(label="physics")
        physics_before_dynamics(state_aux, 0.5 * dt)
        dict_op.copy(state, state_aux)
        tasmania.python.utils.time.Timer.stop(label="physics")

        # compute the dynamics
        tasmania.python.utils.time.Timer.start(label="dynamics")
        state["time"] = nl.init_time + i * dt
        state_prv = dycore(state, {}, dt)
        extension = {key: state[key] for key in state if key not in state_prv}
        state_prv.update(extension)
        state_prv["time"] = nl.init_time + (i + 0.5) * dt
        tasmania.python.utils.time.Timer.stop(label="dynamics")

        # compute the physics
        tasmania.python.utils.time.Timer.start(label="physics")
        physics_after_dynamics(state_prv, 0.5 * dt)
        dict_op.copy(state, state_prv)
        tasmania.python.utils.time.Timer.stop(label="physics")

        # stop timing
        tasmania.python.utils.time.Timer.stop(label="compute_time")

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # print umax and vmax for validation
    u = taz.to_numpy(state["x_velocity_at_u_locations"].data)
    umax = u[:, :-1, :-1].max()
    v = taz.to_numpy(state["y_velocity_at_v_locations"].data)
    vmax = v[:-1, :, :-1].max()
    print(f"Validation: umax = {umax:.5f}, vmax = {vmax:.5f}")

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
