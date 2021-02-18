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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest

from gt4py import gtscript

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.physics.microphysics.kessler import KesslerFallVelocity
from tasmania.python.physics.microphysics.sedimentation_fluxes import (
    FirstOrderUpwind,
    SecondOrderUpwind,
)
from tasmania.python.physics.microphysics.utils import (
    Clipping,
    SedimentationFlux,
    Precipitation,
)
from tasmania.python.utils.storage import get_dataarray_3d
from tasmania.python.utils.backend import is_gt, get_gt_backend

from tests import conf
from tests.strategies import (
    st_one_of,
    st_domain,
    st_isentropic_state_f,
    st_raw_field,
)
from tests.utilities import compare_arrays, compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"
ndpw = "number_density_of_precipitation_water"


def precipitation_validation(state, timestep, maxcfl, rhow):
    rho = to_numpy(state["air_density"].to_units("kg m^-3").data)
    h = to_numpy(state["height_on_interface_levels"].to_units("m").data)
    qr = to_numpy(state[mfpw].to_units("g g^-1").data)
    vt = to_numpy(state["raindrop_fall_velocity"].to_units("m s^-1").data)

    dt = timestep.total_seconds()
    dh = h[:, :, :-1] - h[:, :, 1:]
    ids = np.where(vt > maxcfl * dh / dt)
    vt[ids] = maxcfl * dh[ids] / dt

    return 3.6e6 * rho * qr * vt / rhow


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_clipping(data, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            zaxis_length=(2, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
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
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diagnostics = clip(state)

    assert len(clip.input_properties) == len(names)
    assert len(clip.diagnostic_properties) == len(names)

    for name in names:
        # with subtests.test(name=name):
        assert name in clip.input_properties
        assert name in clip.diagnostic_properties

        q_np = to_numpy(state[name].to_units("g g^-1").data)
        q_np[q_np < 0] = 0

        assert name in diagnostics
        compare_dataarrays(
            get_dataarray_3d(q_np[:nx, :ny, :nz], grid, "g g^-1"),
            diagnostics[name],
            compare_coordinate_values=False,
            slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
        )

    assert len(diagnostics) == len(names)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_precipitation(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            zaxis_length=(2, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=True,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
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
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    state.update(rfv(state))

    comp = Precipitation(
        domain,
        grid_type,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    tendencies, diagnostics = comp(state, timestep)

    assert len(tendencies) == 0

    rho_np = to_numpy(state["air_density"].to_units("kg m^-3").data)[
        :nx, :ny, nz - 1 : nz
    ]
    qr_np = to_numpy(state[mfpw].to_units("g g^-1").data)[
        :nx, :ny, nz - 1 : nz
    ]
    vt_np = to_numpy(state["raindrop_fall_velocity"].to_units("m s^-1").data)[
        :nx, :ny, nz - 1 : nz
    ]
    rhow = comp.rpc["density_of_liquid_water"]
    prec = 3.6e6 * rho_np * qr_np * vt_np / rhow
    assert "precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(prec, grid, "mm hr^-1"),
        diagnostics["precipitation"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    accprec_np = to_numpy(
        state["accumulated_precipitation"].to_units("mm").data
    )[:nx, :ny]
    accprec = accprec_np + timestep.total_seconds() * prec / 3.6e3
    assert "accumulated_precipitation" in diagnostics
    compare_dataarrays(
        get_dataarray_3d(accprec, grid, "mm"),
        diagnostics["accumulated_precipitation"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    assert len(diagnostics) == 2


class WrappingStencil(StencilFactory):
    def __init__(
        self, core, *, backend=None, backend_options=None, storage_options=None
    ):
        super().__init__(backend, backend_options, storage_options)
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "func": core.stencil_subroutine("flux", backend=self.backend),
            "nb": core.nb,
        }
        self.stencil = self.compile("stencil")

    def __call__(self, rho, h, qr, vt, dfdz):
        nx, ny, mk = rho.shape
        self.stencil(
            rho=rho,
            h=h,
            qr=qr,
            vt=vt,
            dfdz=dfdz,
            origin=(0, 0, 0),
            domain=(nx, ny, mk - 1),
        )

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil")
    def stencil_numpy_defs(rho, h, qr, vt, dfdz, *, origin, domain):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kb, ke = origin[2], origin[2] + domain[2]

        dfdz[i, j, kb : kb + nb] = 0.0
        dfdz[i, j, kb + nb : ke] = func(
            rho=rho[i, j, kb:ke],
            h=h[i, j, kb:ke],
            q=qr[i, j, kb:ke],
            vt=vt[i, j, kb:ke],
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil")
    def stencil_gt4py_defs(
        rho: gtscript.Field["dtype"],
        h: gtscript.Field["dtype"],
        qr: gtscript.Field["dtype"],
        vt: gtscript.Field["dtype"],
        dfdz: gtscript.Field["dtype"],
    ):
        from __externals__ import func, nb

        with computation(PARALLEL), interval(0, nb):
            dfdz = 0.0
        with computation(PARALLEL), interval(nb, None):
            dfdz = func(rho=rho, h=h, q=qr, vt=vt)


def first_order_flux_validation(rho, h, qr, vt, staggering=False):
    if staggering:
        tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])
    else:
        tmp_h = h

    out = np.zeros_like(rho)
    out[:, :, :1] = 0.0
    out[:, :, 1:] = (
        rho[:, :, :-1] * qr[:, :, :-1] * vt[:, :, :-1]
        - rho[:, :, 1:] * qr[:, :, 1:] * vt[:, :, 1:]
    ) / (tmp_h[:, :, :-1] - tmp_h[:, :, 1:])

    return out


def second_order_flux_validation(rho, h, qr, vt, staggering=False):
    if staggering:
        tmp_h = 0.5 * (h[:, :, :-1] + h[:, :, 1:])
    else:
        tmp_h = h

    a = np.zeros_like(rho)
    a[:, :, 2:] = (
        2 * tmp_h[:, :, 2:] - tmp_h[:, :, 1:-1] - tmp_h[:, :, :-2]
    ) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:])
        * (tmp_h[:, :, :-2] - tmp_h[:, :, 2:])
    )
    b = np.zeros_like(rho)
    b[:, :, 2:] = (tmp_h[:, :, :-2] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:])
        * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )
    c = np.zeros_like(rho)
    c[:, :, 2:] = -(tmp_h[:, :, 1:-1] - tmp_h[:, :, 2:]) / (
        (tmp_h[:, :, :-2] - tmp_h[:, :, 2:])
        * (tmp_h[:, :, :-2] - tmp_h[:, :, 1:-1])
    )

    out = np.zeros_like(rho)
    out[:, :, :2] = 0.0
    out[:, :, 2:] = (
        a[:, :, 2:] * rho[:, :, 2:] * qr[:, :, 2:] * vt[:, :, 2:]
        + b[:, :, 2:] * rho[:, :, 1:-1] * qr[:, :, 1:-1] * vt[:, :, 1:-1]
        + c[:, :, 2:] * rho[:, :, :-2] * qr[:, :, :-2] * vt[:, :, :-2]
    )

    return out


flux_properties = {
    "first_order_upwind": {
        "type": FirstOrderUpwind,
        "validation": first_order_flux_validation,
    },
    "second_order_upwind": {
        "type": SecondOrderUpwind,
        "validation": second_order_flux_validation,
    },
}


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
@pytest.mark.parametrize(
    "flux_type", ("first_order_upwind", "second_order_upwind")
)
def test_sedimentation_flux(data, backend, dtype, flux_type):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            zaxis_length=(3, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    rho = data.draw(
        st_raw_field(
            storage_shape, 1, 1e4, backend=backend, storage_options=so
        ),
        label="rho",
    )
    h = data.draw(
        st_raw_field(
            storage_shape, 1, 1e4, backend=backend, storage_options=so
        ),
        label="h",
    )
    qr = data.draw(
        st_raw_field(
            storage_shape, 1, 1e4, backend=backend, storage_options=so
        ),
        label="qr",
    )
    vt = data.draw(
        st_raw_field(
            storage_shape, 1, 1e4, backend=backend, storage_options=so
        ),
        label="vt",
    )

    # ========================================
    # test bed
    # ========================================
    dfdz_val = flux_properties[flux_type]["validation"](
        to_numpy(rho), to_numpy(h), to_numpy(qr), to_numpy(vt),
    )

    core = SedimentationFlux.factory(flux_type, backend=backend)
    assert isinstance(core, flux_properties[flux_type]["type"])

    dfdz = zeros(backend, shape=storage_shape, storage_options=so)
    ws = WrappingStencil(
        core, backend=backend, backend_options=bo, storage_options=so
    )
    ws(rho, h, qr, vt, dfdz)

    compare_arrays(dfdz, dfdz_val, slice=(slice(nx), slice(ny), slice(nz)))


def kessler_sedimentation_validation(
    nx, ny, nz, state, timestep, flux_scheme, maxcfl
):
    rho = to_numpy(state["air_density"].to_units("kg m^-3").data)[
        :nx, :ny, :nz
    ]
    h = to_numpy(state["height_on_interface_levels"].to_units("m").data)[
        :nx, :ny, : nz + 1
    ]
    qr = to_numpy(state[mfpw].to_units("g g^-1").data)[:nx, :ny, :nz]
    vt = to_numpy(state["raindrop_fall_velocity"].to_units("m s^-1").data)[
        :nx, :ny, :nz
    ]

    # dt = timestep.total_seconds()
    # dh = h[:, :, :-1] - h[:, :, 1:]
    # ids = np.where(vt > maxcfl * dh / dt)
    # vt[ids] = maxcfl * dh[ids] / dt

    dfdz = flux_properties[flux_scheme]["validation"](
        rho, h, qr, vt, staggering=True
    )

    return dfdz / rho


# @hyp_settings
# @given(data=hyp_st.data())
# @pytest.mark.parametrize("backend", conf.backend)
# @pytest.mark.parametrize("dtype", conf.dtype)
# def _test_kessler_sedimentation(data, backend, dtype):
#     # ========================================
#     # random data generation
#     # ========================================
#     domain = data.draw(st_domain(), label="domain")
#
#     grid_type = data.draw(
#         st_one_of(("physical", "numerical")), label="grid_type"
#     )
#     grid = (
#         domain.physical_grid
#         if grid_type == "physical"
#         else domain.numerical_grid
#     )
#     state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")
#
#     flux_scheme = data.draw(
#         st_one_of(("first_order_upwind", "second_order_upwind")),
#         label="flux_scheme",
#     )
#     maxcfl = data.draw(hyp_st.floats(min_value=0, max_value=1), label="maxcfl")
#
#     timestep = data.draw(
#         hyp_st.timedeltas(
#             min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
#         ),
#         label="timestep",
#     )
#
#     aligned_index = data.draw(st_one_of(conf.aligned_index), label="aligned_index")
#
#     # ========================================
#     # test bed
#     # ========================================
#     dtype = grid.x.dtype
#
#     rfv = KesslerFallVelocity(
#         domain,
#         grid_type,
#         backend=backend,
#         dtype=dtype,
#         aligned_index=aligned_index,
#     )
#     diagnostics = rfv(state)
#     state.update(diagnostics)
#
#     tracer = {mfpw: {"units": "g g^-1", "velocity": "raindrop_fall_velocity"}}
#     sed = Sedimentation(
#         domain,
#         grid_type,
#         tracer,
#         flux_scheme,
#         maxcfl,
#         backend=gt.mode.NUMPY,
#         dtype=dtype,
#     )
#
#     #
#     # test properties
#     #
#     assert "air_density" in sed.input_properties
#     assert "height_on_interface_levels" in sed.input_properties
#     assert mfpw in sed.input_properties
#     assert "raindrop_fall_velocity" in sed.input_properties
#     assert len(sed.input_properties) == 4
#
#     assert mfpw in sed.tendency_properties
#     assert len(sed.tendency_properties) == 1
#
#     assert len(sed.diagnostic_properties) == 0
#
#     #
#     # test numerics
#     #
#     tendencies, diagnostics = sed(state, timestep)
#
#     assert mfpw in tendencies
#     raw_mfpw_val = kessler_sedimentation_validation(
#         state, timestep, flux_scheme, maxcfl
#     )
#     compare_dataarrays(
#         get_dataarray_3d(raw_mfpw_val, grid, "g g^-1 s^-1"),
#         tendencies[mfpw],
#         compare_coordinate_values=False,
#     )
#     assert len(tendencies) == 1
#
#     assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
