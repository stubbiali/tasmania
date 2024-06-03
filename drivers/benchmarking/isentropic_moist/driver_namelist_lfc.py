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
import sys

from sympl._core.time import Timer

from tasmania.domain.domain import Domain
from tasmania.framework.allocators import zeros
from tasmania.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.framework.generic_functions import to_numpy
from tasmania.isentropic.dynamics.dycore import IsentropicDynamicalCore
from tasmania.isentropic.physics.coriolis import IsentropicConservativeCoriolis
from tasmania.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
    IsentropicVelocityComponents,
)
from tasmania.isentropic.physics.horizontal_smoothing import IsentropicHorizontalSmoothing
from tasmania.isentropic.physics.turbulence import IsentropicSmagorinsky
from tasmania.isentropic.physics.vertical_advection import IsentropicVerticalAdvection
from tasmania.isentropic.state import get_isentropic_state_from_brunt_vaisala_frequency
from tasmania.isentropic.utils import AirPotentialTemperatureToDiagnostic
from tasmania.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustmentPrognostic,
    KesslerSedimentation,
)
from tasmania.physics.microphysics.utils import Precipitation
from tasmania.utils.storage import get_dataarray_3d
from tasmania.utils.utils import feed_module
from tasmania.utils.xarrayx import DataArrayDictOperator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import namelist_lfc
from utils import exec_info_to_csv, inject_backend, run_info_to_csv


@click.command()
@click.option("-b", "--backend", type=str, default=None, help="The backend.")
@click.option(
    "-n",
    "--namelist",
    type=str,
    default="namelist_lfc.py",
    help="The namelist file.",
)
@click.option("--no-log", is_flag=True, help="Disable log.")
def main(backend=None, namelist="namelist_lfc.py", no_log=False):
    # ============================================================
    # The namelist
    # ============================================================
    _namelist = namelist.replace("/", ".")
    _namelist = _namelist[:-3] if _namelist.endswith(".py") else _namelist
    exec(f"import {_namelist} as namelist_module")
    nl = locals()["namelist_module"]
    feed_module(target=nl, source=namelist_lfc)
    inject_backend(nl, backend)

    # ============================================================
    # The underlying domain
    # ============================================================
    domain = Domain(
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
    state = get_isentropic_state_from_brunt_vaisala_frequency(
        cgrid,
        nl.init_time,
        nl.x_velocity,
        nl.y_velocity,
        nl.brunt_vaisala,
        moist=True,
        precipitation=nl.sedimentation,
        relative_humidity=nl.relative_humidity,
        backend=nl.backend,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    domain.horizontal_boundary.reference_state = state

    # add tendency_of_air_potential_temperature to the state
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        zeros(backend, shape=storage_shape, storage_options=nl.so),
        cgrid,
        "K s^-1",
        grid_shape=(cgrid.nx, cgrid.ny, cgrid.nz),
        set_coordinates=False,
    )

    # ============================================================
    # The slow tendencies
    # ============================================================
    args = []

    # component calculating the Coriolis acceleration
    cf = IsentropicConservativeCoriolis(
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

    # component implementing the Smagorinsky turbulence model
    turb = IsentropicSmagorinsky(
        domain,
        smagorinsky_constant=nl.smagorinsky_constant,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(turb)

    # component calculating the microphysics
    ke = KesslerMicrophysics(
        domain,
        "numerical",
        air_pressure_on_interface_levels=True,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        rain_evaporation=nl.rain_evaporation,
        autoconversion_threshold=nl.autoconversion_threshold,
        autoconversion_rate=nl.autoconversion_rate,
        collection_rate=nl.collection_rate,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(ke)

    # components calculating the tendencies "emulating" the saturation adjustment
    sa = KesslerSaturationAdjustmentPrognostic(
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        saturation_rate=nl.saturation_rate,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(sa)

    # component promoting air_potential_temperature to state variable
    t2d = AirPotentialTemperatureToDiagnostic(
        domain,
        "numerical",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(t2d)

    if nl.vertical_advection:
        # component integrating the vertical flux
        vf = IsentropicVerticalAdvection(
            domain,
            flux_scheme=nl.vertical_flux_scheme,
            moist=True,
            tendency_of_air_potential_temperature_on_interface_levels=False,
            enable_checks=nl.enable_checks,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(vf)

    # component estimating the raindrop fall velocity
    rfv = KesslerFallVelocity(
        domain,
        "numerical",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(rfv)

    # component integrating the sedimentation flux
    sd = KesslerSedimentation(
        domain,
        "numerical",
        sedimentation_flux_scheme=nl.sedimentation_flux_scheme,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(sd)

    # wrap the components in a ConcurrentCoupling object
    slow_tends = ConcurrentCoupling(
        *args,
        execution_policy="serial",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    # ============================================================
    # The slow diagnostics
    # ============================================================
    args = []

    # component retrieving the diagnostic variables
    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
    idv = IsentropicDiagnostics(
        domain,
        grid_type="numerical",
        moist=True,
        pt=pt,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(idv)

    # component calculating the accumulated precipitation
    ap = Precipitation(
        domain,
        "numerical",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(rfv)
    args.append(ap)

    # component performing the horizontal smoothing
    hs = IsentropicHorizontalSmoothing(
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
    vc = IsentropicVelocityComponents(
        domain,
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(vc)

    # wrap the components in a ConcurrentCoupling object
    slow_diags = ConcurrentCoupling(
        *args,
        execution_policy="serial",
        enable_checks=nl.enable_checks,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    # ============================================================
    # The dynamical core
    # ============================================================
    dycore = IsentropicDynamicalCore(
        domain,
        moist=True,
        # parameterizations
        fast_tendency_component=None,
        fast_diagnostic_component=None,
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
        # horizontal smoothing
        smooth=False,
        smooth_moist=False,
        # backend settings
        enable_checks=nl.enable_checks,
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
    dict_op = DataArrayDictOperator(
        backend=nl.backend, backend_options=nl.bo, storage_options=nl.so
    )

    # warm up caches
    dycore.update_topography(timedelta(seconds=0.0))
    tendencies, diagnostics_before = slow_tends(state, timedelta(seconds=0.0))
    dict_op.update_swap(state, diagnostics_before)
    state_new = dycore(state, tendencies, timedelta(seconds=0.0))
    _, diagnostics_after = slow_diags(state, timedelta(seconds=0.0))
    dict_op.update_swap(state_new, diagnostics_after)

    # reset timers
    Timer.reset()

    for i in range(nt):
        # start timing
        Timer.start(label="compute_time")

        # swap old and new state
        state, state_new = state_new, state

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # calculate the slow tendencies
        slow_tends(
            state,
            dt,
            out_tendencies=tendencies,
            out_diagnostics=diagnostics_before,
        )
        dict_op.update_swap(state, diagnostics_before)

        # step the solution
        dycore(state, tendencies, dt, out_state=state_new)
        state_new["time"] = nl.init_time + (i + 1) * dt

        # retrieve the slow diagnostics
        slow_diags(state_new, dt, out_diagnostics=diagnostics_after)
        dict_op.update_swap(state_new, diagnostics_after)

        # stop timing
        Timer.stop(label="compute_time")

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # print umax and vmax for validation
    u = to_numpy(state["x_velocity_at_u_locations"].data)
    umax = u[:, :-1, :-1].max()
    v = to_numpy(state["y_velocity_at_v_locations"].data)
    vmax = v[:-1, :, :-1].max()
    print(f"Validation: umax = {umax:.5f}, vmax = {vmax:.5f}")

    # print logs
    print(f"Compute time: {Timer.get_time('compute_time', 's'):.3f} s.")
    print(f"Stencil time: {Timer.get_time('stencil', 's'):.3f} s.")

    if not no_log:
        # save to file
        exec_info_to_csv(nl.exec_info_csv, "lfc", nl.backend, nl.bo)
        run_info_to_csv(nl.run_info_csv, "lfc", nl.backend)
        Timer.log(nl.log_txt, "s")


if __name__ == "__main__":
    main()
