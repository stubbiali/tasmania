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
from datetime import timedelta
from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.physics.microphysics.kessler import KesslerFallVelocity
from tasmania.python.physics.microphysics.porz import PorzFallVelocity
from tasmania.python.physics.microphysics.utils import (
    SedimentationFlux,
    _FirstOrderUpwind,
    _SecondOrderUpwind,
    Sedimentation,
    Precipitation,
)
from tasmania.python.utils.storage_utils import get_storage_descriptor
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, halo as conf_halo
    from .utils import (
        compare_arrays,
        compare_dataarrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo
    from utils import (
        compare_arrays,
        compare_dataarrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"
ndpw = "number_density_of_precipitation_water"


def precipitation_validation(state, timestep, maxcfl, rhow):
    rho = state["air_density"].to_units("kg m^-3").values
    h = state["height_on_interface_levels"].to_units("m").values
    qr = state[mfpw].to_units("g g^-1").values
    vt = state["raindrop_fall_velocity"].to_units("m s^-1").values

    dt = timestep.total_seconds()
    dh = h[:, :, :-1] - h[:, :, 1:]
    ids = np.where(vt > maxcfl * dh / dt)
    vt[ids] = maxcfl * dh[ids] / dt

    return 3.6e6 * rho[:, :, -1:] * qr[:, :, -1:] * vt[:, :, -1:] / rhow


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_precipitation(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(
        st_isentropic_state_f(grid, moist=True, precipitation=True), label="state"
    )

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    rfv = KesslerFallVelocity(domain, grid_type, backend=backend, dtype=dtype, halo=halo)
    state.update(rfv(state))

    comp = Precipitation(domain, grid_type, backend=backend, dtype=dtype, halo=halo)

    tendencies, diagnostics = comp(state, timestep)

    assert len(tendencies) == 0

    rho = state["air_density"].to_units("kg m^-3").values[:, :, -1:]
    qr = state[mfpw].to_units("g g^-1").values[:, :, -1:]
    vt = state["raindrop_fall_velocity"].to_units("m s^-1").values[:, :, -1:]
    rhow = comp._pcs["density_of_liquid_water"]
    prec = 3.6e6 * rho * qr * vt / rhow
    assert "precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(prec, grid, "mm hr^-1"),
        diagnostics["precipitation"],
        compare_coordinate_values=False,
    )

    accprec = state["accumulated_precipitation"].to_units("mm").values
    accprec_val = accprec + timestep.total_seconds() * prec / 3.6e3
    assert "accumulated_precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(accprec_val, grid, "mm"),
        diagnostics["accumulated_precipitation"],
        compare_coordinate_values=False,
    )

    assert len(diagnostics) == 2


class WrappingStencil:
    def __init__(self, core, backend, rebuild):
        self.core = core
        decorator = gt.stencil(
            backend, rebuild=rebuild, externals={"core": core.__call__}
        )
        self.stencil = decorator(self.stencil_defs)

    def __call__(self, rho, h, qr, vt, dfdz):
        nx, ny, mk = rho.data.shape
        nb = self.core.nb
        self.stencil(
            rho=rho,
            h=h,
            qr=qr,
            vt=vt,
            dfdz=dfdz,
            origin={"_all_": (0, 0, nb)},
            domain=(nx, ny, mk - nb - 1),
        )

    @staticmethod
    def stencil_defs(
        rho: gt.storage.f64_sd,
        h: gt.storage.f64_sd,
        qr: gt.storage.f64_sd,
        vt: gt.storage.f64_sd,
        dfdz: gt.storage.f64_sd,
    ):
        dfdz = core(rho=rho, h=h, q=qr, vt=vt)


def first_order_flux_validation(rho, h, qr, vt):
    tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

    out = np.zeros_like(rho, dtype=rho.dtype)
    out[:, :, 1:] = (
        rho[:, :, :-1] * qr[:, :, :-1] * vt[:, :, :-1]
        - rho[:, :, 1:] * qr[:, :, 1:] * vt[:, :, 1:]
    ) / (tmp_h[:, :, :-1] - tmp_h[:, :, 1:])

    return out


def second_order_flux_validation(rho, h, qr, vt):
    tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

    a = np.zeros_like(rho, dtype=rho.dtype)
    a[:, :, 2:] = (2 * tmp_h[:, :, 2:] - tmp_h[:, :, 1:-1] - tmp_h[:, :, :-2]) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 2:])
    )
    b = np.zeros_like(rho, dtype=rho.dtype)
    b[:, :, 2:] = (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )
    c = np.zeros_like(rho, dtype=rho.dtype)
    c[:, :, 2:] = -(tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )

    out = np.zeros_like(rho, dtype=rho.dtype)
    out[:, :, 2:] = (
        a[:, :, 2:] * (rho[:, :, 2:] * qr[:, :, 2:] * vt[:, :, 2:])
        + b[:, :, 2:] * (rho[:, :, 1:-1] * qr[:, :, 1:-1] * vt[:, :, 1:-1])
        + c[:, :, 2:] * (rho[:, :, :-2] * qr[:, :, :-2] * vt[:, :, :-2])
    )

    return out


flux_properties = {
    "first_order_upwind": {
        "type": _FirstOrderUpwind,
        "validation": first_order_flux_validation,
    },
    "second_order_upwind": {
        "type": _SecondOrderUpwind,
        "validation": second_order_flux_validation,
    },
}


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_sedimentation_flux(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 20)), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    dtype = grid.x.dtype

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    phi = data.draw(
        st_arrays(
            dtype,
            (nx + 1, ny + 1, nz + 1),
            elements=st_floats(min_value=1, max_value=1e4),
            fill=hyp_st.nothing(),
        ),
        label="rho",
    )

    flux_type = data.draw(
        st_one_of(("first_order_upwind", "second_order_upwind")), label="flux_type"
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    descriptor = get_storage_descriptor((nx, ny, nz + 1), dtype, halo=halo)
    rho = gt.storage.from_array(phi[:-1, :-1], descriptor, backend=backend)
    h = gt.storage.from_array(phi[1:, :-1], descriptor, backend=backend)
    qr = gt.storage.from_array(phi[:-1, 1:], descriptor, backend=backend)
    vt = gt.storage.from_array(phi[1:, 1:], descriptor, backend=backend)
    dfdz = gt.storage.zeros(descriptor, backend=backend)

    core = SedimentationFlux.factory(flux_type)
    assert isinstance(core, flux_properties[flux_type]["type"])
    ws = WrappingStencil(core, backend, True)
    ws(rho, h, qr, vt, dfdz)
    dfdz_val = flux_properties[flux_type]["validation"](
        rho.data[:, :, :nz], h.data, qr.data[:, :, :nz], vt.data[:, :, :nz]
    )
    compare_arrays(dfdz.data[:, :, :nz], dfdz_val)


def kessler_sedimentation_validation(state, timestep, flux_scheme, maxcfl):
    rho = state["air_density"].to_units("kg m^-3").values
    h = state["height_on_interface_levels"].to_units("m").values
    qr = state[mfpw].to_units("g g^-1").values
    vt = state["raindrop_fall_velocity"].to_units("m s^-1").values

    dt = timestep.total_seconds()
    dh = h[:, :, :-1] - h[:, :, 1:]
    # ids = np.where(vt > maxcfl * dh / dt)
    # vt[ids] = maxcfl * dh[ids] / dt

    dfdz = flux_properties[flux_scheme]["validation"](rho, h, qr, vt)

    return dfdz / rho


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_kessler_sedimentation(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    flux_scheme = data.draw(
        st_one_of(("first_order_upwind", "second_order_upwind")), label="flux_scheme"
    )
    maxcfl = data.draw(hyp_st.floats(min_value=0, max_value=1), label="maxcfl")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    rfv = KesslerFallVelocity(domain, grid_type, backend=backend, dtype=dtype, halo=halo)
    diagnostics = rfv(state)
    state.update(diagnostics)

    tracer = {mfpw: {"units": "g g^-1", "velocity": "raindrop_fall_velocity"}}
    sed = Sedimentation(
        domain, grid_type, tracer, flux_scheme, maxcfl, backend=gt.mode.NUMPY, dtype=dtype
    )

    #
    # test properties
    #
    assert "air_density" in sed.input_properties
    assert "height_on_interface_levels" in sed.input_properties
    assert mfpw in sed.input_properties
    assert "raindrop_fall_velocity" in sed.input_properties
    assert len(sed.input_properties) == 4

    assert mfpw in sed.tendency_properties
    assert len(sed.tendency_properties) == 1

    assert len(sed.diagnostic_properties) == 0

    #
    # test numerics
    #
    tendencies, diagnostics = sed(state, timestep)

    assert mfpw in tendencies
    raw_mfpw_val = kessler_sedimentation_validation(state, timestep, flux_scheme, maxcfl)
    compare_dataarrays(
        get_dataarray_3d(raw_mfpw_val, grid, "g g^-1 s^-1"),
        tendencies[mfpw],
        compare_coordinate_values=False,
    )
    assert len(tendencies) == 1

    assert len(diagnostics) == 0


def porz_sedimentation_validation(state, timestep, flux_scheme, maxcfl):
    rho = state["air_density"].to_units("kg m^-3").values
    h = state["height_on_interface_levels"].to_units("m").values
    qr = state[mfpw].to_units("g g^-1").values
    nr = state[ndpw].to_units("kg^-1").values
    vq = state["raindrop_fall_velocity"].to_units("m s^-1").values
    vn = state["number_density_of_raindrop_fall_velocity"].to_units("m s^-1").values

    dt = timestep.total_seconds()
    dh = h[:, :, :-1] - h[:, :, 1:]

    # ids = np.where(vq > maxcfl * dh / dt)
    # vq[ids] = maxcfl * dh[ids] / dt
    # ids = np.where(vn > maxcfl * dh / dt)
    # vn[ids] = maxcfl * dh[ids] / dt

    dfdz_qr = flux_properties[flux_scheme]["validation"](rho, h, qr, vq)
    dfdz_nr = flux_properties[flux_scheme]["validation"](rho, h, nr, vn)

    return dfdz_qr / rho, dfdz_nr / rho


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_porz_sedimentation(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    flux_scheme = data.draw(
        st_one_of(("first_order_upwind", "second_order_upwind")), label="flux_scheme"
    )
    maxcfl = data.draw(hyp_st.floats(min_value=0, max_value=1), label="maxcfl")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    rfv = PorzFallVelocity(domain, grid_type, backend=backend, dtype=dtype)
    diagnostics = rfv(state)
    state.update(diagnostics)

    tracers = {
        mfpw: {"units": "g g^-1", "velocity": "raindrop_fall_velocity"},
        ndpw: {"units": "kg^-1", "velocity": "number_density_of_raindrop_fall_velocity"},
    }
    sed = Sedimentation(
        domain,
        grid_type,
        tracers,
        flux_scheme,
        maxcfl,
        backend=gt.mode.NUMPY,
        dtype=dtype,
    )

    #
    # test properties
    #
    assert "air_density" in sed.input_properties
    assert "height_on_interface_levels" in sed.input_properties
    assert mfpw in sed.input_properties
    assert ndpw in sed.input_properties
    assert "raindrop_fall_velocity" in sed.input_properties
    assert "number_density_of_raindrop_fall_velocity" in sed.input_properties
    assert len(sed.input_properties) == 6

    assert mfpw in sed.tendency_properties
    assert ndpw in sed.tendency_properties
    assert len(sed.tendency_properties) == 2

    assert len(sed.diagnostic_properties) == 0

    #
    # test numerics
    #
    tendencies, diagnostics = sed(state, timestep)

    raw_mfpw_val, raw_ndpw_val = porz_sedimentation_validation(
        state, timestep, flux_scheme, maxcfl
    )

    assert mfpw in tendencies
    compare_dataarrays(
        get_dataarray_3d(raw_mfpw_val, grid, "g g^-1 s^-1"),
        tendencies[mfpw],
        compare_coordinate_values=False,
    )
    assert ndpw in tendencies
    compare_dataarrays(
        get_dataarray_3d(raw_ndpw_val, grid, "kg^-1 s^-1"),
        tendencies[ndpw],
        compare_coordinate_values=False,
    )
    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
