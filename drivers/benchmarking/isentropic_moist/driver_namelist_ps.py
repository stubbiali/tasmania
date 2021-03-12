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

from drivers.benchmarking.isentropic_moist import namelist_ps
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
@click.option(
    "-o", "--output", type=bool, default=True, help="Output.",
)
def main(backend=None, namelist="namelist_ps.py", output=True):
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
        moist=True,
        precipitation=nl.sedimentation,
        relative_humidity=nl.relative_humidity,
        backend=nl.backend,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    domain.horizontal_boundary.reference_state = state

    # add tendency_of_air_potential_temperature to the state
    state["tendency_of_air_potential_temperature"] = taz.get_dataarray_3d(
        taz.zeros(nl.backend, shape=storage_shape, storage_options=nl.so),
        cgrid,
        "K s^-1",
        grid_shape=(cgrid.nx, cgrid.ny, cgrid.nz),
        set_coordinates=False,
    )

    # ============================================================
    # The dynamics
    # ============================================================
    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
    dycore = taz.IsentropicDynamicalCore(
        domain,
        moist=True,
        # parameterizations
        intermediate_tendency_component=None,
        intermediate_diagnostic_component=None,
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
    # The physics
    # ============================================================
    args = []
    ptis = nl.physics_time_integration_scheme

    # component retrieving the diagnostic variables
    idv = taz.IsentropicDiagnostics(
        domain,
        grid_type="numerical",
        moist=True,
        pt=pt,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(taz.TimeIntegrationOptions(component=idv))

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
    args.append(
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
        moist=nl.smooth_moist,
        smooth_moist_coeff=nl.smooth_moist_coeff,
        smooth_moist_coeff_max=nl.smooth_moist_coeff_max,
        smooth_moist_damp_depth=nl.smooth_moist_damp_depth,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(taz.TimeIntegrationOptions(component=hs))

    # component implementing the Smagorinsky turbulence model
    turb = taz.IsentropicSmagorinsky(
        domain,
        nl.smagorinsky_constant,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(
        taz.TimeIntegrationOptions(
            component=turb,
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    # component retrieving the velocity components
    ivc = taz.IsentropicVelocityComponents(
        domain,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(taz.TimeIntegrationOptions(component=ivc))

    # component promoting air_potential_temperature to state variable
    t2d = taz.AirPotentialTemperature2Diagnostic(domain, "numerical")

    # component calculating the microphysics
    ke = taz.KesslerMicrophysics(
        domain,
        "numerical",
        air_pressure_on_interface_levels=True,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        rain_evaporation=nl.rain_evaporation,
        autoconversion_threshold=nl.autoconversion_threshold,
        autoconversion_rate=nl.autoconversion_rate,
        collection_rate=nl.collection_rate,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(
        taz.TimeIntegrationOptions(
            component=taz.ConcurrentCoupling(
                ke,
                t2d,
                execution_policy="serial",
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            ),
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    # component downgrading tendency_of_air_potential_temperature to tendency variable
    d2t = taz.AirPotentialTemperature2Tendency(domain, "numerical")

    # component performing the saturation adjustment
    sa = taz.KesslerSaturationAdjustmentPrognostic(
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        saturation_rate=nl.saturation_rate,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(
        taz.TimeIntegrationOptions(
            component=taz.ConcurrentCoupling(
                d2t,
                sa,
                t2d,
                execution_policy="serial",
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            ),
            scheme=ptis,
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    if nl.vertical_advection:
        if nl.implicit_vertical_advection:
            # component integrating the vertical flux
            vf = taz.IsentropicImplicitVerticalAdvectionDiagnostic(
                domain,
                moist=True,
                tendency_of_air_potential_temperature_on_interface_levels=False,
                backend=nl.backend,
                backend_options=nl.bo,
                storage_shape=storage_shape,
                storage_options=nl.so,
            )
            args.append(taz.TimeIntegrationOptions(component=vf))
        else:
            # component integrating the vertical flux
            vf = taz.IsentropicVerticalAdvection(
                domain,
                flux_scheme=nl.vertical_flux_scheme,
                moist=True,
                tendency_of_air_potential_temperature_on_interface_levels=False,
                backend=nl.backend,
                backend_options=nl.bo,
                storage_shape=storage_shape,
                storage_options=nl.so,
            )
            args.append(
                taz.TimeIntegrationOptions(
                    component=vf,
                    scheme="rk3ws",
                    substeps=1,
                    backend=nl.backend,
                    backend_options=nl.bo,
                    storage_options=nl.so,
                )
            )

    # component estimating the raindrop fall velocity
    rfv = taz.KesslerFallVelocity(
        domain,
        "numerical",
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )

    # component integrating the sedimentation flux
    sd = taz.KesslerSedimentation(
        domain,
        "numerical",
        sedimentation_flux_scheme=nl.sedimentation_flux_scheme,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(
        taz.TimeIntegrationOptions(
            component=taz.ConcurrentCoupling(rfv, sd),
            scheme="rk3ws",
            substeps=1,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_options=nl.so,
        )
    )

    # component calculating the accumulated precipitation
    ap = taz.Precipitation(
        domain,
        "numerical",
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(
        taz.TimeIntegrationOptions(
            component=taz.DiagnosticComponentComposite(
                rfv, ap, execution_policy="serial"
            )
        )
    )

    # wrap the components in a ParallelSplitting object
    physics = taz.ParallelSplitting(
        *args,
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=True,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so
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
        taz.Timer.start(label="compute_time")

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # calculate the dynamics
        taz.Timer.start(label="dynamics")
        state_prv = dycore(state, {}, dt)
        extension = {key: state[key] for key in state if key not in state_prv}
        state_prv.update(extension)
        taz.Timer.stop(label="dynamics")

        # calculate the physics
        taz.Timer.start(label="physics")
        physics(state, state_prv, dt)
        taz.Timer.stop(label="physics")

        # update the state
        dict_op.copy(state, state_prv)

        # stop timing
        taz.Timer.stop(label="compute_time")

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # print logs
    print(f"Compute time: {taz.Timer.get_time('compute_time', 's')} s.")

    if output:
        # save to file
        exec_info_to_csv(nl.exec_info_csv, nl.backend, nl.bo)
        run_info_to_csv(
            nl.run_info_csv, backend, taz.Timer.get_time("compute_time", "s")
        )
        taz.Timer.log(nl.log_txt, "s")


if __name__ == "__main__":
    main()