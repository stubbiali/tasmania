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
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
from property_cached import cached_property
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

from tests import conf
from tests.strategies import (
    st_one_of,
    st_domain,
    st_isentropic_state_f,
    st_raw_field,
)
from tests.suites import (
    DiagnosticComponentTestSuite,
    DomainSuite,
    TendencyComponentTestSuite,
)
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"
ndpw = "number_density_of_precipitation_water"


class ClippingTestSuite(DiagnosticComponentTestSuite):
    def __init__(self, domain_suite):
        self.names = []
        if domain_suite.hyp_data.draw(hyp_st.booleans(), label="if_qv"):
            self.names.append(mfwv)
        if domain_suite.hyp_data.draw(hyp_st.booleans(), label="if_qc"):
            self.names.append(mfcw)
        if domain_suite.hyp_data.draw(hyp_st.booleans(), label="if_qr"):
            self.names.append(mfpw)
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return Clipping(
            self.ds.domain,
            self.ds.grid_type,
            self.names,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=True,
                precipitation=False,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_diagnostics(self, raw_state_np):
        return {
            name: np.where(raw_state_np[name] < 0.0, 0.0, raw_state_np[name])
            for name in self.names
        }


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_clipping(data, backend, dtype, subtests):
    ds = DomainSuite(data, backend, dtype)
    ts = ClippingTestSuite(ds)
    ts.run()


class PrecipitationTestSuite(TendencyComponentTestSuite):
    @cached_property
    def component(self):
        return Precipitation(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def assert_allclose(self, name, field_a, field_b):
        try:
            compare_arrays(
                field_a,
                field_b,
                slice=(
                    slice(self.ds.grid.nx),
                    slice(self.ds.grid.ny),
                    slice(1),
                ),
            )
        except AssertionError:
            raise RuntimeError(f"assert_allclose failed on {name}")

    def get_state(self):
        state = self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=True,
                precipitation=True,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )
        rfv = KesslerFallVelocity(
            self.ds.domain,
            self.ds.grid_type,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )
        state.update(rfv(state))
        return state

    def get_tendencies_and_diagnostics(self, raw_state_np, dt):
        nz = self.ds.grid.nz
        rho = raw_state_np["air_density"][:, :, nz - 1 : nz]
        qr = raw_state_np[mfpw][:, :, nz - 1 : nz]
        vt = raw_state_np["raindrop_fall_velocity"][:, :, nz - 1 : nz]
        in_accprec = raw_state_np["accumulated_precipitation"]
        rhow = self.component.rpc["density_of_liquid_water"]

        prec = np.zeros_like(in_accprec)
        prec[:, :, :1] = 3.6e6 * rho * qr * vt / rhow

        tendencies = {}
        diagnostics = {
            "precipitation": prec,
            "accumulated_precipitation": in_accprec + dt * prec / 3.6e3,
        }

        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_precipitation(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = PrecipitationTestSuite(ds)
    ts.run()


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
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="stencil"
    )
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
        to_numpy(rho),
        to_numpy(h),
        to_numpy(qr),
        to_numpy(vt),
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
