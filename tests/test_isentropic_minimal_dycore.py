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
from copy import deepcopy
from datetime import timedelta
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics as RawIsentropicDiagnostics,
)
from tasmania.python.isentropic.dynamics.minimal_dycore import (
    IsentropicMinimalDynamicalCore,
)
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics
from tasmania.python.isentropic.physics.pressure_gradient import (
    IsentropicConservativePressureGradient,
)

try:
    from .conf import backend as conf_backend  # nb as conf_nb
    from .test_isentropic_horizontal_fluxes import get_fifth_order_upwind_fluxes
    from .test_isentropic_prognostic import forward_euler_step
    from .utils import (
        compare_arrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
    )
except ModuleNotFoundError:
    from conf import backend as conf_backend  # nb as conf_nb
    from test_isentropic_horizontal_fluxes import get_fifth_order_upwind_fluxes
    from test_isentropic_minimal_prognostic import forward_euler_step
    from utils import (
        compare_arrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def rk3ws_stage(
    stage,
    timestep,
    raw_state_now,
    raw_state_int,
    raw_state_ref,
    raw_tendencies,
    raw_state_new,
    field_properties,
    dx,
    dy,
    hb,
    moist,
    hv,
    wc,
    damp,
    vd,
    smooth,
    hs,
):
    u_int = raw_state_int["x_velocity_at_u_locations"]
    v_int = raw_state_int["y_velocity_at_v_locations"]

    if moist:
        wc.get_density_of_water_constituent(
            raw_state_int["air_isentropic_density"],
            raw_state_int[mfwv],
            raw_state_int["isentropic_density_of_water_vapor"],
        )
        wc.get_density_of_water_constituent(
            raw_state_int["air_isentropic_density"],
            raw_state_int[mfcw],
            raw_state_int["isentropic_density_of_cloud_liquid_water"],
        )
        wc.get_density_of_water_constituent(
            raw_state_int["air_isentropic_density"],
            raw_state_int[mfpw],
            raw_state_int["isentropic_density_of_precipitation_water"],
        )

        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_int["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_int["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_int["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    if stage == 0:
        fraction = 1.0 / 3.0
    elif stage == 1:
        fraction = 0.5
    else:
        fraction = 1.0

    raw_state_new["time"] = raw_state_now["time"] + fraction * timestep

    dt = fraction * timestep.total_seconds()

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    for name in names:
        phi_now = raw_state_now[name]
        phi_int = raw_state_int[name]
        phi_tnd = raw_tendencies.get(name, None)
        phi_new = raw_state_new[name]
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            "xy",
            dx,
            dy,
            dt,
            u_int,
            v_int,
            phi_now,
            phi_int,
            phi_tnd,
            phi_new,
        )

    if moist:
        wc.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_water_vapor"],
            raw_state_new[mfwv],
            clipping=True,
        )
        wc.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_cloud_liquid_water"],
            raw_state_new[mfcw],
            clipping=True,
        )
        wc.get_mass_fraction_of_water_constituent_in_air(
            raw_state_new["air_isentropic_density"],
            raw_state_new["isentropic_density_of_precipitation_water"],
            raw_state_new[mfpw],
            clipping=True,
        )

    hb.dmn_enforce_raw(raw_state_new, field_properties=field_properties)

    if damp:
        names = (
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
        )
        for name in names:
            phi_now = raw_state_now[name]
            phi_new = raw_state_new[name]
            phi_ref = raw_state_ref[name]
            phi_out = raw_state_new[name]
            vd(timestep, phi_now, phi_new, phi_ref, phi_out)

    if smooth:
        for name in field_properties:
            phi = raw_state_new[name]
            phi_out = raw_state_new[name]
            hs(phi, phi_out)
            hb.dmn_enforce_field(
                phi_out,
                field_name=name,
                field_units=field_properties[name]["units"],
                time=raw_state_new["time"],
            )

    hv.get_velocity_components(
        raw_state_new["air_isentropic_density"],
        raw_state_new["x_momentum_isentropic"],
        raw_state_new["y_momentum_isentropic"],
        raw_state_new["x_velocity_at_u_locations"],
        raw_state_new["y_velocity_at_v_locations"],
    )
    hb.dmn_set_outermost_layers_x(
        raw_state_new["x_velocity_at_u_locations"],
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=raw_state_new["time"],
    )
    hb.dmn_set_outermost_layers_y(
        raw_state_new["y_velocity_at_v_locations"],
        field_name="y_velocity_at_v_locations",
        field_units="m s^-1",
        time=raw_state_new["time"],
    )


def rk3ws_step(
    domain,
    moist,
    timestep,
    raw_state_0,
    raw_tendencies,
    hv,
    wc,
    damp,
    damp_at_every_stage,
    vd,
    smooth,
    smooth_at_every_stage,
    hs,
):
    grid, hb = domain.numerical_grid, domain.horizontal_boundary
    nx, ny, nz, nb = grid.nx, grid.ny, grid.nz, hb.nb
    dx, dy = grid.dx.to_units("m").values.item(), grid.dy.to_units("m").values.item()
    dtype = grid.x.dtype

    if moist:
        raw_state_0["isentropic_density_of_water_vapor"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )
        raw_state_0["isentropic_density_of_cloud_liquid_water"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )
        raw_state_0["isentropic_density_of_precipitation_water"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )

    raw_state_1 = deepcopy(raw_state_0)
    raw_state_2 = deepcopy(raw_state_0)
    raw_state_3 = deepcopy(raw_state_0)

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).values
        )

    # stage 0
    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3ws_stage(
        0,
        timestep,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        _damp,
        vd,
        _smooth,
        hs,
    )

    # stage 1
    rk3ws_stage(
        1,
        timestep,
        raw_state_0,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        _damp,
        vd,
        _smooth,
        hs,
    )

    # stage 2
    rk3ws_stage(
        2,
        timestep,
        raw_state_0,
        raw_state_2,
        raw_state_ref,
        raw_tendencies,
        raw_state_3,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        damp,
        vd,
        smooth,
        hs,
    )

    return raw_state_3


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test1(data):
    """
	- Slow tendencies: no
	- Intermediate tendencies: no
	- Intermediate diagnostics: no
	- Sub-stepping: no
	"""
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3), label="domain"
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")
    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(hyp_st.booleans(), label="damp_at_every_stage")

    smooth = data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="smooth_damp_depth"
    )
    smooth_at_every_stage = data.draw(hyp_st.booleans(), label="smooth_at_every_stage")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    hv = HorizontalVelocity(grid, True, backend, dtype)
    wc = WaterConstituent(grid, backend, dtype)
    vd = VerticalDamping.factory(
        "rayleigh", (nx, ny, nz), grid, damp_depth, 0.0002, "s", backend, dtype
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        (nx, ny, nz),
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend,
        dtype,
    )

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        intermediate_tendencies=None,
        intermediate_diagnostics=None,
        fast_tendencies=None,
        fast_diagnostics=None,
        moist=moist,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        dtype=dtype,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 8
    else:
        assert len(dycore.input_properties) == 5

    assert "air_isentropic_density" in dycore.tendency_properties
    assert "x_momentum_isentropic" in dycore.tendency_properties
    assert "y_momentum_isentropic" in dycore.tendency_properties
    if moist:
        assert mfwv in dycore.tendency_properties
        assert mfwv in dycore.tendency_properties
        assert mfpw in dycore.tendency_properties
        assert len(dycore.tendency_properties) == 6
    else:
        assert len(dycore.tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy(state)

    state_new = dycore(state, {}, timestep)

    for key in state:
        if key == "time":
            assert state["time"] == state_dc["time"]
        else:
            assert np.allclose(state[key], state_dc[key])

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_now = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_now[name] = state[name].to_units(props["units"]).values

    raw_state_new_val = rk3ws_step(
        domain,
        moist,
        timestep,
        raw_state_now,
        {},
        hv,
        wc,
        damp,
        damp_at_every_stage,
        vd,
        smooth,
        smooth_at_every_stage,
        hs,
    )

    for name in state_new:
        if name != "time":
            compare_arrays(state_new[name].values, raw_state_new_val[name])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test2(data):
    """
	- Slow tendencies: yes
	- Intermediate tendencies: no
	- Intermediate diagnostics: no
	- Sub-stepping: no
	"""
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3), label="domain"
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")
    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy(state["air_isentropic_density"])
        tendencies["air_isentropic_density"].attrs["units"] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy(state["x_momentum_isentropic"])
        tendencies["x_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy(state["y_momentum_isentropic"])
        tendencies["y_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(hyp_st.booleans(), label="damp_at_every_stage")

    smooth = data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="smooth_damp_depth"
    )
    smooth_at_every_stage = data.draw(hyp_st.booleans(), label="smooth_at_every_stage")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    hv = HorizontalVelocity(grid, True, backend, dtype)
    wc = WaterConstituent(grid, backend, dtype)
    vd = VerticalDamping.factory(
        "rayleigh", (nx, ny, nz), grid, damp_depth, 0.0002, "s", backend, dtype
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        (nx, ny, nz),
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend,
        dtype,
    )

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        intermediate_tendencies=None,
        intermediate_diagnostics=None,
        fast_tendencies=None,
        fast_diagnostics=None,
        moist=moist,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        dtype=dtype,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 8
    else:
        assert len(dycore.input_properties) == 5

    assert "air_isentropic_density" in dycore.tendency_properties
    assert "x_momentum_isentropic" in dycore.tendency_properties
    assert "y_momentum_isentropic" in dycore.tendency_properties
    if moist:
        assert mfwv in dycore.tendency_properties
        assert mfwv in dycore.tendency_properties
        assert mfpw in dycore.tendency_properties
        assert len(dycore.tendency_properties) == 6
    else:
        assert len(dycore.tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key], state_dc[key])

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_now = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_now[name] = state[name].to_units(props["units"]).values

    raw_tendencies = {}
    for name, props in dycore.tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = tendencies[name].to_units(props["units"]).values

    raw_state_new_val = rk3ws_step(
        domain,
        moist,
        timestep,
        raw_state_now,
        raw_tendencies,
        hv,
        wc,
        damp,
        damp_at_every_stage,
        vd,
        smooth,
        smooth_at_every_stage,
        hs,
    )

    for name in state_new:
        if name != "time":
            compare_arrays(state_new[name].values, raw_state_new_val[name])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test3(data):
    """
	- Slow tendencies: yes
	- Intermediate tendencies: yes
	- Intermediate diagnostics: yes
	- Sub-stepping: no
	"""
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3), label="domain"
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")
    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy(state["air_isentropic_density"])
        tendencies["air_isentropic_density"].attrs["units"] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy(state["x_momentum_isentropic"])
        tendencies["x_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy(state["y_momentum_isentropic"])
        tendencies["y_momentum_isentropic"].attrs["units"] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(hyp_st.booleans(), label="damp_at_every_stage")

    smooth = data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="smooth_damp_depth"
    )
    smooth_at_every_stage = data.draw(hyp_st.booleans(), label="smooth_at_every_stage")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    hv = HorizontalVelocity(grid, True, backend, dtype)
    wc = WaterConstituent(grid, backend, dtype)
    vd = VerticalDamping.factory(
        "rayleigh", (nx, ny, nz), grid, damp_depth, 0.0002, "s", backend, dtype
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        (nx, ny, nz),
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend,
        dtype,
    )

    pg = IsentropicConservativePressureGradient(domain, "second_order", backend, dtype)

    dv = IsentropicDiagnostics(
        domain,
        "numerical",
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend,
        dtype,
    )

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        intermediate_tendencies=pg,
        intermediate_diagnostics=dv,
        fast_tendencies=None,
        fast_diagnostics=None,
        moist=moist,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        dtype=dtype,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.tendency_properties
    assert "x_momentum_isentropic" in dycore.tendency_properties
    assert "y_momentum_isentropic" in dycore.tendency_properties
    if moist:
        assert mfwv in dycore.tendency_properties
        assert mfwv in dycore.tendency_properties
        assert mfpw in dycore.tendency_properties
        assert len(dycore.tendency_properties) == 6
    else:
        assert len(dycore.tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "air_pressure_on_interface_levels" in dycore.output_properties
    assert "exner_function_on_interface_levels" in dycore.output_properties
    assert "height_on_interface_levels" in dycore.output_properties
    assert "montgomery_potential" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert "air_density" in dycore.output_properties
        assert "air_temperature" in dycore.output_properties
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 14
    else:
        assert len(dycore.output_properties) == 9

    #
    # test numerics
    #
    state_dc = deepcopy(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        if key == "time":
            assert state["time"] == state_dc["time"]
        else:
            assert np.allclose(state[key], state_dc[key])

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "air_pressure_on_interface_levels" in state_new
    assert "exner_function_on_interface_levels" in state_new
    assert "height_on_interface_levels" in state_new
    assert "montgomery_potential" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert "air_density" in state_new
        assert "air_temperature" in state_new
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 15
    else:
        assert len(state_new) == 10

    raw_state_now = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_now[name] = state[name].to_units(props["units"]).values
    for name, props in dycore.output_properties.items():
        if name not in dycore.input_properties:
            raw_state_now[name] = state[name].to_units(props["units"]).values
    if moist:
        raw_state_now["isentropic_density_of_water_vapor"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )
        raw_state_now["isentropic_density_of_cloud_liquid_water"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )
        raw_state_now["isentropic_density_of_precipitation_water"] = np.zeros(
            (nx, ny, nz), dtype=dtype
        )

    raw_tendencies = {}
    for name, props in dycore.tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = tendencies[name].to_units(props["units"]).values

    raw_tendencies_dc = deepcopy(raw_tendencies)

    if "x_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["x_momentum_isentropic"] = np.zeros((nx, ny, nz), dtype=dtype)
    if "y_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["y_momentum_isentropic"] = np.zeros((nx, ny, nz), dtype=dtype)

    dx, dy = grid.dx.to_units("m").values.item(), grid.dy.to_units("m").values.item()

    raw_state_0 = deepcopy(raw_state_now)
    raw_state_1 = deepcopy(raw_state_now)
    raw_state_2 = deepcopy(raw_state_now)

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).values
        )

    rdv = RawIsentropicDiagnostics(grid, backend=backend, dtype=dtype)

    #
    # stage 0
    #
    s = raw_state_now["air_isentropic_density"]
    mtg = raw_state_now["montgomery_potential"]
    raw_tendencies["x_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[2:, 1:-1] - mtg[:-2, 1:-1]
    ) / (2.0 * dx) + (
        raw_tendencies_dc["x_momentum_isentropic"][1:-1, 1:-1]
        if "x_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )
    raw_tendencies["y_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[1:-1, 2:] - mtg[1:-1, :-2]
    ) / (2.0 * dy) + (
        raw_tendencies_dc["y_momentum_isentropic"][1:-1, 1:-1]
        if "y_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3ws_stage(
        0,
        timestep,
        raw_state_now,
        raw_state_now,
        raw_state_ref,
        raw_tendencies,
        raw_state_0,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        _damp,
        vd,
        _smooth,
        hs,
    )

    rdv.get_diagnostic_variables(
        raw_state_0["air_isentropic_density"],
        raw_state_0["air_pressure_on_interface_levels"][0, 0, 0],
        raw_state_0["air_pressure_on_interface_levels"],
        raw_state_0["exner_function_on_interface_levels"],
        raw_state_0["montgomery_potential"],
        raw_state_0["height_on_interface_levels"],
    )
    if moist:
        rdv.get_air_density(
            raw_state_0["air_isentropic_density"],
            raw_state_0["height_on_interface_levels"],
            raw_state_0["air_density"],
        )
        rdv.get_air_temperature(
            raw_state_0["exner_function_on_interface_levels"],
            raw_state_0["air_temperature"],
        )

    #
    # stage 1
    #
    s = raw_state_0["air_isentropic_density"]
    mtg = raw_state_0["montgomery_potential"]
    raw_tendencies["x_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[2:, 1:-1] - mtg[:-2, 1:-1]
    ) / (2.0 * dx) + (
        raw_tendencies_dc["x_momentum_isentropic"][1:-1, 1:-1]
        if "x_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )
    raw_tendencies["y_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[1:-1, 2:] - mtg[1:-1, :-2]
    ) / (2.0 * dy) + (
        raw_tendencies_dc["y_momentum_isentropic"][1:-1, 1:-1]
        if "y_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3ws_stage(
        1,
        timestep,
        raw_state_now,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        _damp,
        vd,
        _smooth,
        hs,
    )

    rdv.get_diagnostic_variables(
        raw_state_1["air_isentropic_density"],
        raw_state_1["air_pressure_on_interface_levels"][0, 0, 0],
        raw_state_1["air_pressure_on_interface_levels"],
        raw_state_1["exner_function_on_interface_levels"],
        raw_state_1["montgomery_potential"],
        raw_state_1["height_on_interface_levels"],
    )
    if moist:
        rdv.get_air_density(
            raw_state_1["air_isentropic_density"],
            raw_state_1["height_on_interface_levels"],
            raw_state_1["air_density"],
        )
        rdv.get_air_temperature(
            raw_state_1["exner_function_on_interface_levels"],
            raw_state_1["air_temperature"],
        )

    #
    # stage 2
    #
    s = raw_state_1["air_isentropic_density"]
    mtg = raw_state_1["montgomery_potential"]
    raw_tendencies["x_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[2:, 1:-1] - mtg[:-2, 1:-1]
    ) / (2.0 * dx) + (
        raw_tendencies_dc["x_momentum_isentropic"][1:-1, 1:-1]
        if "x_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )
    raw_tendencies["y_momentum_isentropic"][1:-1, 1:-1] = -s[1:-1, 1:-1] * (
        mtg[1:-1, 2:] - mtg[1:-1, :-2]
    ) / (2.0 * dy) + (
        raw_tendencies_dc["y_momentum_isentropic"][1:-1, 1:-1]
        if "y_momentum_isentropic" in raw_tendencies_dc
        else 0.0
    )

    rk3ws_stage(
        2,
        timestep,
        raw_state_now,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        dx,
        dy,
        hb,
        moist,
        hv,
        wc,
        damp,
        vd,
        smooth,
        hs,
    )

    rdv.get_diagnostic_variables(
        raw_state_2["air_isentropic_density"],
        raw_state_2["air_pressure_on_interface_levels"][0, 0, 0],
        raw_state_2["air_pressure_on_interface_levels"],
        raw_state_2["exner_function_on_interface_levels"],
        raw_state_2["montgomery_potential"],
        raw_state_2["height_on_interface_levels"],
    )
    if moist:
        rdv.get_air_density(
            raw_state_2["air_isentropic_density"],
            raw_state_2["height_on_interface_levels"],
            raw_state_2["air_density"],
        )
        rdv.get_air_temperature(
            raw_state_2["exner_function_on_interface_levels"],
            raw_state_2["air_temperature"],
        )

    for name in state_new:
        if name != "time":
            assert np.allclose(
                state_new[name].values, raw_state_2[name], equal_nan=True
            )


if __name__ == "__main__":
    pytest.main([__file__])
