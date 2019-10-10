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
from datetime import datetime
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.isentropic.physics.vertical_advection import (
    IsentropicVerticalAdvection,
    PrescribedSurfaceHeating,
)
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .test_isentropic_minimal_vertical_fluxes import (
        get_upwind_flux,
        get_centered_flux,
        get_third_order_upwind_flux,
        get_fifth_order_upwind_flux,
    )
    from .utils import (
        st_domain,
        st_floats,
        st_isentropic_state_f,
        st_one_of,
        st_raw_field,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from test_isentropic_minimal_vertical_fluxes import (
        get_upwind_flux,
        get_centered_flux,
        get_third_order_upwind_flux,
        get_fifth_order_upwind_flux,
    )
    from utils import st_domain, st_floats, st_isentropic_state_f, st_one_of, st_raw_field


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def set_lower_layers_first_order(nb, dz, w, phi, out):
    wm = w if w.shape[2] == phi.shape[2] else 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    out[:, :, -nb:] = (
        1
        / dz
        * (
            wm[:, :, -nb - 1 : -1] * phi[:, :, -nb - 1 : -1]
            - wm[:, :, -nb:] * phi[:, :, -nb:]
        )
    )


def set_lower_layers_second_order(nb, dz, w, phi, out):
    wm = w if w.shape[2] == phi.shape[2] else 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    out[:, :, -nb:] = (
        0.5
        * (
            - 3.0 * wm[:, :, -nb:] * phi[:, :, -nb:]
            + 4.0 * wm[:, :, -nb - 1 : -1] * phi[:, :, -nb - 1 : -1]
            - wm[:, :, -nb - 2: -2] * phi[:, :, -nb - 2: -2]
        )
        / dz
    )


flux_properties = {
    "upwind": {
        "nb": 1,
        "get_flux": get_upwind_flux,
        "set_lower_layers": set_lower_layers_first_order,
    },
    "centered": {
        "nb": 1,
        "get_flux": get_centered_flux,
        "set_lower_layers": set_lower_layers_second_order,
    },
    "third_order_upwind": {
        "nb": 2,
        "get_flux": get_third_order_upwind_flux,
        "set_lower_layers": set_lower_layers_second_order,
    },
    "fifth_order_upwind": {
        "nb": 3,
        "get_flux": get_fifth_order_upwind_flux,
        "set_lower_layers": set_lower_layers_second_order,
    },
}


def validation(domain, flux_scheme, moist, toaptoil, backend, halo, rebuild, state):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()
    dtype = grid.z.dtype

    nb = flux_properties[flux_scheme]["nb"]
    get_flux = flux_properties[flux_scheme]["get_flux"]
    set_lower_layers = flux_properties[flux_scheme]["set_lower_layers"]

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicVerticalAdvection(
        domain,
        flux_scheme,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=toaptoil,
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=rebuild,
        storage_shape=storage_shape,
    )

    input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if toaptoil:
        input_names.append("tendency_of_air_potential_temperature_on_interface_levels")
    else:
        input_names.append("tendency_of_air_potential_temperature")
    if moist:
        input_names.append(mfwv)
        input_names.append(mfcw)
        input_names.append(mfpw)

    output_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        output_names.append(mfwv)
        output_names.append(mfcw)
        output_names.append(mfpw)

    for name in input_names:
        assert name in fluxer.input_properties
    assert len(fluxer.input_properties) == len(input_names)

    for name in output_names:
        assert name in fluxer.tendency_properties
    assert len(fluxer.tendency_properties) == len(output_names)

    assert fluxer.diagnostic_properties == {}

    if toaptoil:
        name = "tendency_of_air_potential_temperature_on_interface_levels"
        w = state[name].to_units("K s^-1").values[:nx, :ny, : nz + 1]
        w_hl = w
    else:
        name = "tendency_of_air_potential_temperature"
        w = state[name].to_units("K s^-1").values[:nx, :ny, :nz]
        w_hl = np.zeros((nx, ny, nz + 1), dtype=dtype)
        w_hl[:, :, 1:-1] = 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
    su = (
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    sv = (
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    if moist:
        qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
        sqv = s * qv
        qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqc = s * qc
        qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqr = s * qr

    tendencies, diagnostics = fluxer(state)

    out = np.zeros((nx, ny, nz), dtype=dtype)
    up = slice(nb, nz - nb)
    down = slice(nb + 1, nz - nb + 1)

    flux = get_flux(w_hl, s)
    out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
    set_lower_layers(nb, dz, w, s, out)
    assert "air_isentropic_density" in tendencies
    assert np.allclose(
        out, tendencies["air_isentropic_density"][:nx, :ny, :nz], equal_nan=True
    )

    flux = get_flux(w_hl, su)
    out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
    set_lower_layers(nb, dz, w, su, out)
    assert "x_momentum_isentropic" in tendencies
    assert np.allclose(
        out, tendencies["x_momentum_isentropic"][:nx, :ny, :nz], equal_nan=True
    )

    flux = get_flux(w_hl, sv)
    out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
    set_lower_layers(nb, dz, w, sv, out)
    assert "y_momentum_isentropic" in tendencies
    assert np.allclose(
        out, tendencies["y_momentum_isentropic"][:nx, :ny, :nz], equal_nan=True
    )

    if moist:
        flux = get_flux(w_hl, sqv)
        out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
        set_lower_layers(nb, dz, w, sqv, out)
        out /= s
        assert mfwv in tendencies
        assert np.allclose(out, tendencies[mfwv][:nx, :ny, :nz], equal_nan=True)

        flux = get_flux(w_hl, sqc)
        out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
        set_lower_layers(nb, dz, w, sqc, out)
        out /= s
        assert mfcw in tendencies
        assert np.allclose(out, tendencies[mfcw][:nx, :ny, :nz], equal_nan=True)

        flux = get_flux(w_hl, sqr)
        out[:, :, nb:-nb] = -(flux[:, :, up] - flux[:, :, down]) / dz
        set_lower_layers(nb, dz, w, sqr, out)
        out /= s
        assert mfpw in tendencies
        assert np.allclose(out, tendencies[mfpw][:nx, :ny, :nz], equal_nan=True)

    assert len(tendencies) == len(output_names)

    assert diagnostics == {}


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 20)), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype, halo=halo),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(domain, "upwind", False, False, backend, halo, True, state)
    validation(domain, "upwind", False, True, backend, halo, True, state)
    validation(domain, "upwind", True, False, backend, halo, True, state)
    validation(domain, "upwind", True, True, backend, halo, True, state)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
@reproduce_failure('4.28.0', b'AXicY2BAAYIzGBhRBJiZULi8nQyDGvz//18aQ+S/PABPmAh5')
def test_centered(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 20)), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype, halo=halo),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(domain, "centered", False, False, backend, halo, True, state)
    validation(domain, "centered", False, True, backend, halo, True, state)
    validation(domain, "centered", True, False, backend, halo, True, state)
    validation(domain, "centered", True, True, backend, halo, True, state)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_third_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(5, 20)), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype, halo=halo),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(domain, "third_order_upwind", False, False, backend, halo, True, state)
    validation(domain, "third_order_upwind", False, True, backend, halo, True, state)
    validation(domain, "third_order_upwind", True, False, backend, halo, True, state)
    validation(domain, "third_order_upwind", True, True, backend, halo, True, state)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_fifth_order_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(7, 20)), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype, halo=halo),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    # ========================================
    # test bed
    # ========================================
    validation(domain, "fifth_order_upwind", False, False, backend, halo, True, state)
    validation(domain, "fifth_order_upwind", False, True, backend, halo, True, state)
    validation(domain, "fifth_order_upwind", True, False, backend, halo, True, state)
    validation(domain, "fifth_order_upwind", True, True, backend, halo, True, state)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_prescribed_surface_heating(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid = domain.numerical_grid

    time = data.draw(
        hyp_st.datetimes(min_value=datetime(1992, 2, 20), max_value=datetime(2010, 7, 21))
    )
    field = data.draw(
        st_arrays(
            grid.x.dtype,
            (grid.nx, grid.ny, grid.nz + 1),
            elements=st_floats(min_value=1, max_value=1e4),
            fill=hyp_st.nothing(),
        )
    )
    state = {
        "time": time,
        "air_pressure": get_dataarray_3d(field[:, :, : grid.nz], grid, "Pa"),
        "air_pressure_on_interface_levels": get_dataarray_3d(field, grid, "Pa"),
        "height_on_interface_levels": get_dataarray_3d(field, grid, "m"),
    }

    f0d_sw = data.draw(st_floats(min_value=0, max_value=100))
    f0d_fw = data.draw(st_floats(min_value=0, max_value=100))
    f0n_sw = data.draw(st_floats(min_value=0, max_value=100))
    f0n_fw = data.draw(st_floats(min_value=0, max_value=100))
    w_sw = data.draw(st_floats(min_value=0.1, max_value=100))
    w_fw = data.draw(st_floats(min_value=0.1, max_value=100))
    ad = data.draw(st_floats(min_value=0, max_value=100))
    an = data.draw(st_floats(min_value=0, max_value=100))
    cl = data.draw(st_floats(min_value=0, max_value=100))
    t0 = data.draw(
        hyp_st.datetimes(min_value=datetime(1992, 2, 20), max_value=datetime(2010, 7, 21))
    )

    backend = data.draw(st_one_of(conf_backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    f0d_sw_da = DataArray(f0d_sw, attrs={"units": "W m^-2"})
    f0d_fw_da = DataArray(f0d_fw, attrs={"units": "W m^-2"})
    f0n_sw_da = DataArray(f0n_sw, attrs={"units": "W m^-2"})
    f0n_fw_da = DataArray(f0n_fw, attrs={"units": "W m^-2"})
    w_sw_da = DataArray(w_sw, attrs={"units": "hr^-1"})
    w_fw_da = DataArray(w_fw, attrs={"units": "hr^-1"})
    ad_da = DataArray(ad, attrs={"units": "m^-1"})
    an_da = DataArray(an, attrs={"units": "m^-1"})
    cl_da = DataArray(cl, attrs={"units": "m"})

    rd = 287.0
    cp = 1004.0

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.x.dtype

    x1d = grid.x.to_units("m").values
    y1d = grid.y.to_units("m").values
    x = np.tile(x1d[:, np.newaxis, np.newaxis], (1, ny, 1))
    y = np.tile(y1d[np.newaxis, :, np.newaxis], (nx, 1, 1))

    dt = (state["time"] - t0).total_seconds() / 3600.0

    t = state["time"].hour * 3600 + state["time"].minute * 60 + state["time"].second
    t_sw = 2 * np.pi / w_sw * 3600
    isday = int(t / t_sw) % 2 == 0
    f0_sw = f0d_sw if isday else f0n_sw
    f0_fw = f0d_fw if isday else f0n_fw
    a = ad if isday else an

    #
    # tendency_of_air_potential_temperature_on_interface_levels=False
    # air_pressure_on_interface_levels=False
    #
    if state["time"] > t0:
        theta = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
        p = state["air_pressure"].values
        z = 0.5 * (
            state["height_on_interface_levels"].values[:, :, :-1]
            + state["height_on_interface_levels"].values[:, :, 1:]
        )
        h = state["height_on_interface_levels"].values[:, :, -1:]

        out = (
            theta
            * rd
            * a
            / (p * cp)
            * np.exp(-a * (z - h))
            * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
            * (x ** 2 + y ** 2 < cl ** 2)
        )
    else:
        out = np.zeros((nx, ny, nz), dtype=dtype)

    # tendency_of_air_potential_temperature_in_diagnostics=False
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=False,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "air_potential_temperature" in psf.tendency_properties
    tendencies, diagnostics = psf(state)
    assert "air_potential_temperature" in tendencies
    assert np.allclose(out, tendencies["air_potential_temperature"], equal_nan=True)
    assert diagnostics == {}

    # tendency_of_air_potential_temperature_in_diagnostics=True
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=True,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=False,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
    tendencies, diagnostics = psf(state)
    assert tendencies == {}
    assert "tendency_of_air_potential_temperature" in diagnostics
    assert np.allclose(
        out, diagnostics["tendency_of_air_potential_temperature"], equal_nan=True
    )

    #
    # tendency_of_air_potential_temperature_on_interface_levels=True
    # air_pressure_on_interface_levels=False
    #
    # tendency_of_air_potential_temperature_in_diagnostics=False
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=True,
        air_pressure_on_interface_levels=False,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "air_potential_temperature" in psf.tendency_properties
    tendencies, diagnostics = psf(state)
    assert "air_potential_temperature" in tendencies
    assert np.allclose(out, tendencies["air_potential_temperature"], equal_nan=True)
    assert diagnostics == {}

    # tendency_of_air_potential_temperature_in_diagnostics=True
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=True,
        tendency_of_air_potential_temperature_on_interface_levels=True,
        air_pressure_on_interface_levels=False,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
    tendencies, diagnostics = psf(state)
    assert tendencies == {}
    assert "tendency_of_air_potential_temperature" in diagnostics
    assert np.allclose(
        out, diagnostics["tendency_of_air_potential_temperature"], equal_nan=True
    )

    #
    # tendency_of_air_potential_temperature_on_interface_levels=False
    # air_pressure_on_interface_levels=True
    #
    if state["time"] > t0:
        theta = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
        p = 0.5 * (
            state["air_pressure_on_interface_levels"].values[:, :, :-1]
            + state["air_pressure_on_interface_levels"].values[:, :, 1:]
        )
        z = 0.5 * (
            state["height_on_interface_levels"].values[:, :, :-1]
            + state["height_on_interface_levels"].values[:, :, 1:]
        )
        h = state["height_on_interface_levels"].values[:, :, -1:]

        out = (
            theta
            * rd
            * a
            / (p * cp)
            * np.exp(-a * (z - h))
            * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
            * (x ** 2 + y ** 2 < cl ** 2)
        )
    else:
        out = np.zeros((nx, ny, nz), dtype=dtype)

    # tendency_of_air_potential_temperature_in_diagnostics=False
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure_on_interface_levels" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "air_potential_temperature" in psf.tendency_properties
    tendencies, diagnostics = psf(state)
    assert "air_potential_temperature" in tendencies
    assert np.allclose(out, tendencies["air_potential_temperature"], equal_nan=True)
    assert diagnostics == {}

    # tendency_of_air_potential_temperature_in_diagnostics=True
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=True,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure_on_interface_levels" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "tendency_of_air_potential_temperature" in psf.diagnostic_properties
    tendencies, diagnostics = psf(state)
    assert tendencies == {}
    assert "tendency_of_air_potential_temperature" in diagnostics
    assert np.allclose(
        out, diagnostics["tendency_of_air_potential_temperature"], equal_nan=True
    )

    #
    # tendency_of_air_potential_temperature_on_interface_levels=True
    # air_pressure_on_interface_levels=True
    #
    if state["time"] > t0:
        theta = grid.z_on_interface_levels.to_units("K").values[np.newaxis, np.newaxis, :]
        p = state["air_pressure_on_interface_levels"].values
        z = state["height_on_interface_levels"].values
        h = state["height_on_interface_levels"].values[:, :, -1:]

        out = (
            theta
            * rd
            * a
            / (p * cp)
            * np.exp(-a * (z - h))
            * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
            * (x ** 2 + y ** 2 < cl ** 2)
        )
    else:
        out = np.zeros((nx, ny, nz + 1), dtype=dtype)

    # tendency_of_air_potential_temperature_in_diagnostics=False
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=True,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure_on_interface_levels" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert "air_potential_temperature_on_interface_levels" in psf.tendency_properties
    tendencies, diagnostics = psf(state)
    assert "air_potential_temperature_on_interface_levels" in tendencies
    assert np.allclose(
        out, tendencies["air_potential_temperature_on_interface_levels"], equal_nan=True
    )
    assert diagnostics == {}

    # tendency_of_air_potential_temperature_in_diagnostics=True
    psf = PrescribedSurfaceHeating(
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=True,
        tendency_of_air_potential_temperature_on_interface_levels=True,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=f0d_sw_da,
        amplitude_at_day_fw=f0d_fw_da,
        amplitude_at_night_sw=f0n_sw_da,
        amplitude_at_night_fw=f0n_fw_da,
        frequency_sw=w_sw_da,
        frequency_fw=w_fw_da,
        attenuation_coefficient_at_day=ad_da,
        attenuation_coefficient_at_night=an_da,
        characteristic_length=cl_da,
        starting_time=t0,
        backend=backend,
    )
    assert "air_pressure_on_interface_levels" in psf.input_properties
    assert "height_on_interface_levels" in psf.input_properties
    assert (
        "tendency_of_air_potential_temperature_on_interface_levels"
        in psf.diagnostic_properties
    )
    tendencies, diagnostics = psf(state)
    assert tendencies == {}
    assert "tendency_of_air_potential_temperature_on_interface_levels" in diagnostics
    assert np.allclose(
        out,
        diagnostics["tendency_of_air_potential_temperature_on_interface_levels"],
        equal_nan=True,
    )


if __name__ == "__main__":
    # pytest.main([__file__])
    test_centered()
