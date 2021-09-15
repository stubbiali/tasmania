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

from tasmania.python.framework.allocators import zeros
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
    IsentropicImplicitVerticalAdvectionPrognostic,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_isentropic_state_f, st_raw_field
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def setup_tridiagonal_system(gamma, w, phi, a=None, b=None, c=None, d=None):
    ni, nj, nk = phi.shape

    a = np.zeros_like(phi) if a is None else a
    b = np.zeros_like(phi) if b is None else b
    c = np.zeros_like(phi) if c is None else c
    d = np.zeros_like(phi) if d is None else d

    for i in range(ni):
        for j in range(nj):
            a[i, j, 0] = 0.0
            b[i, j, 0] = 1.0
            c[i, j, 0] = 0.0
            d[i, j, 0] = phi[i, j, 0]

            for k in range(1, nk - 1):
                a[i, j, k] = gamma * w[i, j, k - 1]
                b[i, j, k] = 1.0
                c[i, j, k] = -gamma * w[i, j, k + 1]
                d[i, j, k] = phi[i, j, k] - gamma * (
                    w[i, j, k - 1] * phi[i, j, k - 1]
                    - w[i, j, k + 1] * phi[i, j, k + 1]
                )

            a[i, j, nk - 1] = 0.0
            b[i, j, nk - 1] = 1.0
            c[i, j, nk - 1] = 0.0
            d[i, j, nk - 1] = phi[i, j, nk - 1]

    return a, b, c, d


def thomas_validation(a, b, c, d, x=None):
    nx, ny, nz = a.shape

    w = np.zeros_like(b)
    beta = b.copy()
    delta = d.copy()
    for i in range(nx):
        for j in range(ny):
            w[i, j, 0] = 0.0
            for k in range(1, nz):
                w[i, j, k] = (
                    a[i, j, k] / beta[i, j, k - 1]
                    if beta[i, j, k - 1] != 0.0
                    else a[i, j, k]
                )
                beta[i, j, k] = b[i, j, k] - w[i, j, k] * c[i, j, k - 1]
                delta[i, j, k] = d[i, j, k] - w[i, j, k] * delta[i, j, k - 1]

    x = np.zeros_like(b) if x is None else x
    for i in range(nx):
        for j in range(ny):
            x[i, j, -1] = (
                delta[i, j, -1] / beta[i, j, -1]
                if beta[i, j, -1] != 0.0
                else delta[i, j, -1] / b[i, j, -1]
            )
            for k in range(nz - 2, -1, -1):
                x[i, j, k] = (
                    (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1])
                    / beta[i, j, k]
                    if beta[i, j, k] != 0.0
                    else (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1])
                    / b[i, j, k]
                )

    return x


class IsentropicImplicitVerticalAdvectionDiagnosticTestSuite(
    TendencyComponentTestSuite
):
    def __init__(self, domain_suite, moist, toaptoil):
        self.moist = moist
        self.toaptoil = toaptoil
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return IsentropicImplicitVerticalAdvectionDiagnostic(
            self.ds.domain,
            self.moist,
            self.toaptoil,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz

        state = self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=self.moist,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

        field = self.hyp_data.draw(
            st_raw_field(
                self.ds.storage_shape or (nx, ny, nz + 1),
                -1e4,
                1e4,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="field",
        )
        if self.toaptoil:
            state[
                "tendency_of_air_potential_temperature_on_interface_levels"
            ] = get_dataarray_3d(
                field,
                self.ds.grid,
                "K s^-1",
                grid_shape=(nx, ny, nz + 1),
                set_coordinates=False,
            )
        else:
            state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
                field if self.ds.storage_shape else field[:nx, :ny, :nz],
                self.ds.grid,
                "K s^-1",
                grid_shape=(nx, ny, nz),
                set_coordinates=False,
            )

        return state

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        dz = self.ds.grid.dz.to_units("K").values.item()
        gamma = dt / (4.0 * dz)

        if self.toaptoil:
            w_hl = raw_state_np[
                "tendency_of_air_potential_temperature_on_interface_levels"
            ]
            w = zeros("numpy", shape=(nx, ny, nz), storage_options=self.ds.so)
            w[...] = 0.5 * (w_hl[:nx, :ny, :nz] + w_hl[:nx, :ny, 1 : nz + 1])
        else:
            w = raw_state_np["tendency_of_air_potential_temperature"]

        diagnostics = {}

        s = raw_state_np["air_isentropic_density"][:nx, :ny, :nz]
        a, b, c, d = setup_tridiagonal_system(gamma, w, s)
        out_s = thomas_validation(a, b, c, d)
        diagnostics["air_isentropic_density"] = out_s

        su = raw_state_np["x_momentum_isentropic"][:nx, :ny, :nz]
        a, b, c, d = setup_tridiagonal_system(gamma, w, su)
        diagnostics["x_momentum_isentropic"] = thomas_validation(a, b, c, d)

        sv = raw_state_np["y_momentum_isentropic"][:nx, :ny, :nz]
        a, b, c, d = setup_tridiagonal_system(gamma, w, sv)
        diagnostics["y_momentum_isentropic"] = thomas_validation(a, b, c, d)

        if self.moist:
            qv = raw_state_np[mfwv][:nx, :ny, :nz]
            sqv = s * qv
            a, b, c, d = setup_tridiagonal_system(gamma, w, sqv)
            out_sqv = thomas_validation(a, b, c, d)
            diagnostics[mfwv] = out_sqv / out_s

            qc = raw_state_np[mfcw][:nx, :ny, :nz]
            sqc = s * qc
            a, b, c, d = setup_tridiagonal_system(gamma, w, sqc)
            out_sqc = thomas_validation(a, b, c, d)
            diagnostics[mfcw] = out_sqc / out_s

            qr = raw_state_np[mfpw][:nx, :ny, :nz]
            sqr = s * qr
            a, b, c, d = setup_tridiagonal_system(gamma, w, sqr)
            out_sqr = thomas_validation(a, b, c, d)
            diagnostics[mfpw] = out_sqr / out_s

        return {}, diagnostics

    def assert_allclose(self, name, field_a, field_b):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        slc = (slice(0, nx), slice(0, ny), slice(0, nz))
        try:
            compare_arrays(field_a, field_b, slice=slc)
        except AssertionError:
            raise RuntimeError(f"assert_allclose failed on {name}")


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_diagnostic(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(
        data, backend, dtype, grid_type="numerical", zaxis_length=(3, 50)
    )
    ts = IsentropicImplicitVerticalAdvectionDiagnosticTestSuite(
        ds, moist=False, toaptoil=False
    )
    ts.run()
    ts = IsentropicImplicitVerticalAdvectionDiagnosticTestSuite(
        ds, moist=True, toaptoil=False
    )
    ts.run()
    ts = IsentropicImplicitVerticalAdvectionDiagnosticTestSuite(
        ds, moist=False, toaptoil=True
    )
    ts.run()
    ts = IsentropicImplicitVerticalAdvectionDiagnosticTestSuite(
        ds, moist=True, toaptoil=True
    )
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
    # test_diagnostic("numpy", float)
