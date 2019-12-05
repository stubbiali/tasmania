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
from hypothesis import given, HealthCheck, settings, strategies as hyp_st
import numpy as np
import pytest

from gt4py import gtscript, storage as gt_storage

from tasmania.python.physics.microphysics.kessler import KesslerFallVelocity
from tasmania.python.physics.microphysics.utils import (
    Clipping,
    SedimentationFlux,
    _FirstOrderUpwind,
    _SecondOrderUpwind,
    Sedimentation,
    Precipitation,
)
from tasmania.python.utils.storage_utils import zeros
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, default_origin as conf_dorigin
    from .utils import (
        compare_arrays,
        compare_dataarrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
        st_raw_field,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, default_origin as conf_dorigin
    from utils import (
        compare_arrays,
        compare_dataarrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
        st_raw_field,
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

    return 3.6e6 * rho * qr * vt / rhow


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_clipping(data):
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(2, 20)), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    names = []
    if data.draw(hyp_st.booleans(), label="if_qv"):
        names.append(mfwv)
    if data.draw(hyp_st.booleans(), label="if_qc"):
        names.append(mfcw)
    if data.draw(hyp_st.booleans(), label="if_qr"):
        names.append(mfpw)

    # ========================================
    # test bed
    # ========================================
    clip = Clipping(
        domain,
        grid_type,
        names,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
        rebuild=False,
    )

    diagnostics = clip(state)

    for name in names:
        assert name in clip.input_properties
        assert name in clip.diagnostic_properties
    assert len(clip.input_properties) == len(names)
    assert len(clip.diagnostic_properties) == len(names)

    for name in names:
        q = state[name].to_units('g g^-1').values
        q[q < 0] = 0

        assert name in diagnostics
        compare_dataarrays(
            get_dataarray_3d(q[:nx, :ny, :nz], grid, "g g^-1"),
            diagnostics[name][:nx, :ny, :nz],
            compare_coordinate_values=False,
        )

    assert len(diagnostics) == len(names)


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
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(2, 20)), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    rfv = KesslerFallVelocity(
        domain,
        grid_type,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    state.update(rfv(state))

    comp = Precipitation(
        domain,
        grid_type,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )

    tendencies, diagnostics = comp(state, timestep)

    assert len(tendencies) == 0

    rho = state["air_density"].to_units("kg m^-3").values[:nx, :ny, nz - 1 : nz]
    qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, nz - 1 : nz]
    vt = state["raindrop_fall_velocity"].to_units("m s^-1").values[:nx, :ny, nz - 1 : nz]
    rhow = comp._pcs["density_of_liquid_water"]
    prec = 3.6e6 * rho * qr * vt / rhow
    assert "precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(prec, grid, "mm hr^-1"),
        diagnostics["precipitation"][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )

    accprec = state["accumulated_precipitation"].to_units("mm").values[:nx, :ny]
    accprec_val = accprec + timestep.total_seconds() * prec / 3.6e3
    assert "accumulated_precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(accprec_val, grid, "mm"),
        diagnostics["accumulated_precipitation"][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )

    assert len(diagnostics) == 2


class WrappingStencil:
    def __init__(self, core, backend, rebuild):
        self.core = core
        decorator = gtscript.stencil(
            backend,
            name=core.__class__.__name__,
            rebuild=rebuild,
            externals={"core": core.__call__, "extent": core.nb},
        )
        self.stencil = decorator(self.stencil_defs)

    def __call__(self, rho, h, qr, vt, dfdz):
        nx, ny, mk = rho.shape
        self.stencil(
            rho=rho,
            h=h,
            qr=qr,
            vt=vt,
            dfdz=dfdz,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, mk - 1),
        )

    @staticmethod
    def stencil_defs(
        rho: gtscript.Field[np.float64],
        h: gtscript.Field[np.float64],
        qr: gtscript.Field[np.float64],
        vt: gtscript.Field[np.float64],
        dfdz: gtscript.Field[np.float64],
    ):
        from __externals__ import core, extent

        with computation(PARALLEL), interval(0, extent):
            dfdz = 0.0

        with computation(PARALLEL), interval(extent, None):
            dfdz = core(rho=rho, h=h, q=qr, vt=vt)


def first_order_flux_validation(rho, h, qr, vt):
    tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

    out = deepcopy(rho)
    out[:, :, :1] = 0.0
    out[:, :, 1:] = (
        rho[:, :, :-1] * qr[:, :, :-1] * vt[:, :, :-1]
        - rho[:, :, 1:] * qr[:, :, 1:] * vt[:, :, 1:]
    ) / (tmp_h[:, :, :-1] - tmp_h[:, :, 1:])

    return out


def second_order_flux_validation(rho, h, qr, vt):
    tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])

    a = deepcopy(rho)
    a[:, :, 2:] = (2 * tmp_h[:, :, 2:] - tmp_h[:, :, 1:-1] - tmp_h[:, :, :-2]) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 2:])
    )
    b = deepcopy(rho)
    b[:, :, 2:] = (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )
    c = deepcopy(rho)
    c[:, :, 2:] = -(tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )

    out = deepcopy(rho)
    out[:, :, :2] = 0.0
    out[:, :, 2:] = (
        a[:, :, 2:] * rho[:, :, 2:] * qr[:, :, 2:] * vt[:, :, 2:]
        + b[:, :, 2:] * rho[:, :, 1:-1] * qr[:, :, 1:-1] * vt[:, :, 1:-1]
        + c[:, :, 2:] * rho[:, :, :-2] * qr[:, :, :-2] * vt[:, :, :-2]
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
    gt_storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 20)), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    rho = data.draw(
        st_raw_field(
            storage_shape,
            1,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="rho",
    )
    h = data.draw(
        st_raw_field(
            storage_shape,
            1,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="h",
    )
    qr = data.draw(
        st_raw_field(
            storage_shape,
            1,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="qr",
    )
    vt = data.draw(
        st_raw_field(
            storage_shape,
            1,
            1e4,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="vt",
    )

    flux_type = data.draw(st_one_of(("first_order_upwind",)), label="flux_type")

    # ========================================
    # test bed
    # ========================================
    dfdz = zeros(storage_shape, backend, dtype, default_origin=default_origin)

    core = SedimentationFlux.factory(flux_type)
    assert isinstance(core, flux_properties[flux_type]["type"])

    ws = WrappingStencil(core, backend, True)
    ws(rho, h, qr, vt, dfdz)

    dfdz_val = flux_properties[flux_type]["validation"](
        rho[:nx, :ny, :nz], h[:nx, :ny, : nz + 1], qr[:nx, :ny, :nz], vt[:nx, :ny, :nz]
    )
    # compare_arrays(dfdz[:nx, :ny, :nz], dfdz_val)


def kessler_sedimentation_validation(nx, ny, nz, state, timestep, flux_scheme, maxcfl):
    rho = state["air_density"].to_units("kg m^-3").values[:nx, :ny, :nz]
    h = state["height_on_interface_levels"].to_units("m").values[:nx, :ny, : nz + 1]
    qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
    vt = state["raindrop_fall_velocity"].to_units("m s^-1").values[:nx, :ny, :nz]

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
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    rfv = KesslerFallVelocity(
        domain, grid_type, backend=backend, dtype=dtype, default_origin=default_origin
    )
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


if __name__ == "__main__":
    pytest.main([__file__])
