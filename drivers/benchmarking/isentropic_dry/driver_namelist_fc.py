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
import tasmania as taz
import tasmania.python.utils.time

from drivers.benchmarking.isentropic_dry import namelist_fc
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
    default="namelist_fc.py",
    help="The namelist file.",
)
@click.option(
    "-o", "--output", type=bool, default=True, help="Output.",
)
def main(backend=None, namelist="namelist_fc.py", output=True):
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
    cgrid = domain.numerical_grid
    storage_shape = (cgrid.nx + 1, cgrid.ny + 1, cgrid.nz + 1)

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
        backend=nl.backend,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    domain.horizontal_boundary.reference_state = state

    # ============================================================
    # The intermediate tendencies
    # ============================================================
    args = []

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
    args.append(cf)

    # component implementing the Smagorinsky turbulence model
    turb = taz.IsentropicSmagorinsky(
        domain,
        nl.smagorinsky_constant,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(turb)

    # wrap the components in a ConcurrentCoupling object
    inter_tends = taz.ConcurrentCoupling(
        *args,
        execution_policy="serial",
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so
    )

    # ============================================================
    # The intermediate diagnostics
    # ============================================================
    # component retrieving the diagnostic variables
    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
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

    # ============================================================
    # The slow diagnostics
    # ============================================================
    args = []

    # component performing the horizontal smoothing
    hs = taz.IsentropicHorizontalSmoothing(
        domain,
        nl.smooth_type,
        nl.smooth_coeff,
        nl.smooth_coeff_max,
        nl.smooth_damp_depth,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(hs)

    # component calculating the velocity components
    vc = taz.IsentropicVelocityComponents(
        domain,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(vc)

    # wrap the components in a ConcurrentCoupling object
    slow_diags = taz.DiagnosticComponentComposite(
        *args, execution_policy="serial"
    )

    # ============================================================
    # The dynamical core
    # ============================================================
    dycore = taz.IsentropicDynamicalCore(
        domain,
        moist=False,
        # parameterizations
        intermediate_tendency_component=inter_tends,
        intermediate_diagnostic_component=dv,
        substeps=nl.substeps,
        fast_tendency_component=None,
        fast_diagnostic_component=None,
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

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # calculate the dynamics
        state_new = dycore(state, {}, dt)

        # update the state
        dict_op.copy(state, state_new)

        # calculate the slow physics
        diagnostics = slow_diags(state, dt)
        dict_op.copy(state, diagnostics, unshared_variables_in_output=True)

        # stop timing
        tasmania.python.utils.time.Timer.stop(label="compute_time")

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # print logs
    print(
        f"Compute time: {tasmania.python.utils.time.Timer.get_time('compute_time', 's')} s."
    )

    if output:
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