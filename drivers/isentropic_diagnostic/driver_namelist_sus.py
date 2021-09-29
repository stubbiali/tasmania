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
import os
import sympl
import time

import tasmania as taz

from drivers.benchmarking.utils import inject_backend
from drivers.isentropic_diagnostic import namelist_sus
from drivers.isentropic_diagnostic.utils import print_info


@click.command()
@click.option("-b", "--backend", type=str, default=None, help="The backend.")
@click.option(
    "-n",
    "--namelist",
    type=str,
    default="namelist_us.py",
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
        ngrid,
        "K s^-1",
        grid_shape=(ngrid.nx, ngrid.ny, ngrid.nz),
        set_coordinates=False,
    )

    # ============================================================
    # The dynamics
    # ============================================================
    pt = sympl.DataArray(
        state["air_pressure_on_interface_levels"].data[0, 0, 0],
        attrs={"units": "Pa"},
    )
    dycore = taz.IsentropicDynamicalCore(
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
    dv = taz.IsentropicDiagnostics(
        domain,
        grid_type="numerical",
        moist=True,
        pt=pt,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    args.append(taz.TimeIntegrationOptions(component=dv))

    if nl.coriolis:
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
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )

    if nl.smooth:
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

    if nl.diff:
        # component calculating tendencies due to numerical diffusion
        hd = taz.IsentropicHorizontalDiffusion(
            domain,
            nl.diff_type,
            nl.diff_coeff,
            nl.diff_coeff_max,
            nl.diff_damp_depth,
            moist=nl.diff_moist,
            diffusion_moist_coeff=nl.diff_moist_coeff,
            diffusion_moist_coeff_max=nl.diff_moist_coeff_max,
            diffusion_moist_damp_depth=nl.diff_moist_damp_depth,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(
            taz.TimeIntegrationOptions(
                component=hd,
                scheme=ptis,
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )

    if nl.turbulence:
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
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )

    if nl.coriolis or nl.smooth or nl.diff or nl.turbulence:
        # component retrieving the velocity components
        ivc = taz.IsentropicVelocityComponents(
            domain,
            backend=nl.backend,
            backend_options=nl.bo,
            storage_shape=storage_shape,
            storage_options=nl.so,
        )
        args.append(taz.TimeIntegrationOptions(component=ivc))

    # component clipping the negative values of the water species
    water_species_names = (
        "mass_fraction_of_water_vapor_in_air",
        "mass_fraction_of_cloud_liquid_water_in_air",
        "mass_fraction_of_precipitation_water_in_air",
    )
    # clp = taz.Clipping(domain, "numerical", water_species_names)

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
    if nl.update_frequency > 0:
        from sympl import UpdateFrequencyWrapper

        comp = UpdateFrequencyWrapper(ke, nl.update_frequency * nl.timestep)
    else:
        comp = ke
    if nl.rain_evaporation:
        args.append(
            taz.TimeIntegrationOptions(
                component=taz.ConcurrentCoupling(
                    comp,
                    t2d,
                    execution_policy="serial",
                    backend=nl.backend,
                    backend_options=nl.bo,
                    storage_options=nl.so,
                ),
                scheme=ptis,
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )
    else:
        args.append(
            taz.TimeIntegrationOptions(
                component=comp,
                scheme=ptis,
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )

    # component downgrading tendency_of_air_potential_temperature to tendency variable
    d2t = taz.AirPotentialTemperature2Tendency(domain, "numerical")

    # component performing the saturation adjustment
    sa = taz.KesslerSaturationAdjustmentDiagnostic(
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        backend=nl.backend,
        backend_options=nl.bo,
        storage_shape=storage_shape,
        storage_options=nl.so,
    )
    if nl.rain_evaporation:
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
                backend=nl.backend,
                backend_options=nl.bo,
                storage_options=nl.so,
            )
        )
    else:
        args.append(
            taz.TimeIntegrationOptions(
                component=taz.ConcurrentCoupling(
                    sa,
                    t2d,
                    execution_policy="serial",
                    backend=nl.backend,
                    backend_options=nl.bo,
                    storage_options=nl.so,
                ),
                scheme=ptis,
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
            args.append(
                taz.TimeIntegrationOptions(
                    component=taz.DiagnosticComponentComposite(vf)
                )
            )
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
                    backend=nl.backend,
                    backend_options=nl.bo,
                    storage_options=nl.so,
                )
            )

    if nl.sedimentation:
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
                component=taz.ConcurrentCoupling(
                    rfv,
                    sd,
                    execution_policy="serial",
                    backend=nl.backend,
                    backend_options=nl.bo,
                    storage_options=nl.so,
                ),
                scheme="rk3ws",
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

    # wrap the components in a SequentialUpdateSplitting object
    physics = taz.SequentialUpdateSplitting(*args)

    # ============================================================
    # A NetCDF monitor
    # ============================================================
    if nl.save and nl.filename is not None:
        if os.path.exists(nl.filename):
            os.remove(nl.filename)

        netcdf_monitor = taz.NetCDFMonitor(
            nl.filename,
            domain,
            "physical",
            store_names=nl.store_names,
            backend="numpy",
            storage_options=nl.so,
        )
        netcdf_monitor.store(state)

    # ============================================================
    # Time-marching
    # ============================================================
    dt = nl.timestep
    nt = nl.niter

    # dict operator
    dict_op = taz.DataArrayDictOperator(
        backend=nl.backend,
        backend_options=nl.bo,
        storage_options=nl.so,
    )

    for i in range(nt):
        # start timing
        taz.Timer.start(label="compute_time")

        # update the (time-dependent) topography
        dycore.update_topography((i + 1) * dt)

        # compute the dynamics
        state_prv = dycore(state, {}, dt)
        extension = {key: state[key] for key in state if key not in state_prv}
        state_prv.update(extension)
        state_prv["time"] = nl.init_time + i * dt

        # compute the physics
        physics(state_prv, dt)

        # update the driving state
        dict_op.copy(state, state_prv)

        # stop timing
        taz.Timer.stop(label="compute_time")

        # print useful info
        print_info(dt, i, nl, pgrid, state)

        # shortcuts
        to_save = (
            nl.save
            and (nl.filename is not None)
            and (
                # ((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0))
                i + 1 in nl.save_iterations
                or i + 1 == nt
            )
        )

        if to_save:
            # save the solution
            netcdf_monitor.store(state)

    print("Simulation successfully completed. HOORAY!")

    # ============================================================
    # Post-processing
    # ============================================================
    # dump the solution to file
    if nl.save and nl.filename is not None:
        netcdf_monitor.write()

    # print logs
    print(f"Compute time: {taz.Timer.get_time('compute_time', 's'):.3f} s.")


if __name__ == "__main__":
    main()
