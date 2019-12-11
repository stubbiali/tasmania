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
import argparse
import gt4py as gt
import os
import tasmania as taz
import time

try:
    from .utils import print_info
except (ImportError, ModuleNotFoundError):
    from utils import print_info


gt.storage.prepare_numpy()

# ============================================================
# The namelist
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    metavar="NAMELIST",
    type=str,
    default="namelist_ps.py",
    help="The namelist file.",
    dest="namelist",
)
args = parser.parse_args()
namelist = args.namelist.replace("/", ".")
namelist = namelist[:-3] if namelist.endswith(".py") else namelist
exec("import {} as namelist".format(namelist))
nl = locals()["namelist"]

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
    backend=nl.gt_kwargs["backend"],
    dtype=nl.gt_kwargs["dtype"],
)
pgrid = domain.physical_grid
cgrid = domain.numerical_grid
nl.gt_kwargs["storage_shape"] = (cgrid.nx + 1, cgrid.ny + 1, cgrid.nz + 1)

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
    backend=nl.gt_kwargs["backend"],
    dtype=nl.gt_kwargs["dtype"],
    default_origin=nl.gt_kwargs["default_origin"],
    storage_shape=nl.gt_kwargs["storage_shape"],
    managed_memory=nl.gt_kwargs["managed_memory"],
)
domain.horizontal_boundary.reference_state = state

# add tendency_of_air_potential_temperature to the state
state["tendency_of_air_potential_temperature"] = taz.get_dataarray_3d(
    taz.zeros(
        nl.gt_kwargs["storage_shape"],
        nl.gt_kwargs["backend"],
        nl.gt_kwargs["dtype"],
        default_origin=nl.gt_kwargs["default_origin"],
        managed_memory=nl.gt_kwargs["managed_memory"],
    ),
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
    intermediate_tendencies=None,
    intermediate_diagnostics=None,
    substeps=nl.substeps,
    fast_tendencies=None,
    fast_diagnostics=None,
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
    # gt4py settings
    gt_powered=True,
    **nl.gt_kwargs
)

# ============================================================
# The physics
# ============================================================
args = []
ptis = nl.physics_time_integration_scheme

# component retrieving the diagnostic variables
idv = taz.IsentropicDiagnostics(
    domain, grid_type="numerical", moist=True, pt=pt, **nl.gt_kwargs
)
args.append({"component": idv})

if nl.coriolis:
    # component calculating the Coriolis acceleration
    cf = taz.IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        coriolis_parameter=nl.coriolis_parameter,
        **nl.gt_kwargs
    )
    args.append(
        {
            "component": cf,
            "time_integrator": ptis,
            "gt_powered": nl.gt_powered,
            "time_integrator_kwargs": nl.gt_kwargs,
            "substeps": 1,
        }
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
        **nl.gt_kwargs
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
        moist=nl.diff_moist,
        diffusion_moist_coeff=nl.diff_moist_coeff,
        diffusion_moist_coeff_max=nl.diff_moist_coeff_max,
        diffusion_moist_damp_depth=nl.diff_moist_damp_depth,
        **nl.gt_kwargs
    )
    args.append(
        {
            "component": hd,
            "time_integrator": ptis,
            "gt_powered": nl.gt_powered,
            "time_integrator_kwargs": nl.gt_kwargs,
            "substeps": 1,
        }
    )

if nl.turbulence:
    # component implementing the Smagorinsky turbulence model
    turb = taz.IsentropicSmagorinsky(domain, nl.smagorinsky_constant, **nl.gt_kwargs)
    args.append(
        {
            "component": turb,
            "time_integrator": ptis,
            "gt_powered": nl.gt_powered,
            "time_integrator_kwargs": nl.gt_kwargs,
            "substeps": 1,
        }
    )

if nl.coriolis or nl.smooth or nl.diff or nl.turbulence:
    # component retrieving the velocity components
    ivc = taz.IsentropicVelocityComponents(domain, **nl.gt_kwargs)
    args.append({"component": ivc})

# component downgrading tendency_of_air_potential_temperature to tendency variable
d2t = taz.AirPotentialTemperature2Tendency(domain, "numerical")

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
    saturation_vapor_pressure_formula=nl.saturation_vapor_pressure_formula,
    **nl.gt_kwargs
)

if nl.update_frequency > 0:
    from sympl import UpdateFrequencyWrapper

    comp = UpdateFrequencyWrapper(ke, nl.update_frequency * nl.timestep)
else:
    comp = ke

args.append(
    {
        "component": taz.ConcurrentCoupling(
            d2t,
            comp,
            t2d,
            execution_policy="serial",
            gt_powered=nl.gt_powered,
            **nl.gt_kwargs
        ),
        "time_integrator": ptis,
        "gt_powered": nl.gt_powered,
        "time_integrator_kwargs": nl.gt_kwargs,
        "substeps": 1,
    }
)

if nl.vertical_advection:
    # component integrating the vertical flux
    vf = taz.IsentropicVerticalAdvection(
        domain,
        flux_scheme=nl.vertical_flux_scheme,
        moist=True,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        **nl.gt_kwargs
    )
    args.append(
        {
            "component": vf,
            "time_integrator": "rk3ws",
            "gt_powered": nl.gt_powered,
            "time_integrator_kwargs": nl.gt_kwargs,
            "substeps": 1,
        }
    )

if nl.sedimentation:
    # component estimating the raindrop fall velocity
    rfv = taz.KesslerFallVelocity(domain, "numerical", **nl.gt_kwargs)

    # component integrating the sedimentation flux
    sd = taz.KesslerSedimentation(
        domain,
        "numerical",
        sedimentation_flux_scheme=nl.sedimentation_flux_scheme,
        **nl.gt_kwargs
    )
    args.append(
        {
            "component": taz.ConcurrentCoupling(rfv, sd),
            "time_integrator": ptis,
            "gt_powered": nl.gt_powered,
            "time_integrator_kwargs": nl.gt_kwargs,
            "substeps": 1,
        }
    )

    # component calculating the accumulated precipitation
    ap = taz.Precipitation(domain, "numerical", **nl.gt_kwargs)
    args.append(
        {
            "component": taz.DiagnosticComponentComposite(
                rfv, ap, execution_policy="serial"
            )
        }
    )

# component clipping the negative values of the water species
water_species_names = (
    "mass_fraction_of_water_vapor_in_air",
    "mass_fraction_of_cloud_liquid_water_in_air",
    "mass_fraction_of_precipitation_water_in_air",
)
clp = taz.Clipping(domain, "numerical", water_species_names)
# args.append({"component": clp})

# component performing the saturation adjustment
sa = taz.KesslerSaturationAdjustment(
    domain,
    grid_type="numerical",
    air_pressure_on_interface_levels=True,
    saturation_vapor_pressure_formula=nl.saturation_vapor_pressure_formula,
    **nl.gt_kwargs
)
args.append(
    {
        "component": taz.DiagnosticComponentComposite(
            taz.ConcurrentCoupling(sa, t2d), execution_policy="serial"
        )
    }
)

# wrap the components in a ParallelSplitting object
physics = taz.ParallelSplitting(
    *args,
    execution_policy="serial",
    retrieve_diagnostics_from_provisional_state=True,
    gt_powered=nl.gt_powered,
    **nl.gt_kwargs
)

# ============================================================
# A NetCDF monitor
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

wall_time_start = time.time()
compute_time = 0.0

# dict operator
dict_op = taz.DataArrayDictOperator(nl.gt_powered, **nl.gt_kwargs)

for i in range(nt):
    compute_time_start = time.time()

    # update the (time-dependent) topography
    dycore.update_topography((i + 1) * dt)

    # calculate the dynamics
    state_prv = dycore(state, {}, dt)
    extension = {key: state[key] for key in state if key not in state_prv}
    state_prv.update(extension)
    # state_prv["accumulated_precipitation"] = state["accumulated_precipitation"]

    # calculate the physics
    physics(state, state_prv, dt)

    # update the state
    dict_op.copy(state, state_prv)

    compute_time += time.time() - compute_time_start

    # print useful info
    print_info(dt, i, nl, pgrid, state)

    # shortcuts
    to_save = (
        nl.save
        and (nl.filename is not None)
        and (
            ((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0))
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

# stop chronometer
wall_time = time.time() - wall_time_start

# print logs
print("Total wall time: {}.".format(taz.get_time_string(wall_time)))
print("Compute time: {}.".format(taz.get_time_string(compute_time)))
