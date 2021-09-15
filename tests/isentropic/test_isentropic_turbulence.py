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

from tasmania.python.isentropic.physics.turbulence import IsentropicSmagorinsky

from tests import conf
from tests.strategies import st_isentropic_state_f
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


class IsentropicSmagorinskyTestSuite(TendencyComponentTestSuite):
    @cached_property
    def component(self):
        cs = self.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=10), label="cs"
        )
        return IsentropicSmagorinsky(
            self.ds.domain,
            smagorinsky_constant=cs,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        cs = self.component._cs
        dx = self.ds.grid.dx.to_units("m").values.item()
        dy = self.ds.grid.dy.to_units("m").values.item()
        s = raw_state_np["air_isentropic_density"]
        su = raw_state_np["x_momentum_isentropic"]
        sv = raw_state_np["y_momentum_isentropic"]

        su_tnd = np.zeros_like(su)
        sv_tnd = np.zeros_like(sv)

        u = su / s
        v = sv / s
        s00 = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
        s01 = 0.5 * (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
            + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
        )
        s11 = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
        nu = (
            (cs ** 2)
            * (dx * dy)
            * (2.0 * s00 ** 2 + 4.0 * s01 ** 2 + 2.0 * s11 ** 2) ** 0.5
        )
        u_tnd = 2.0 * (
            (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2])
            / (2.0 * dy)
        )
        v_tnd = 2.0 * (
            (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2])
            / (2.0 * dy)
        )
        su_tnd[2:-2, 2:-2] = s[2:-2, 2:-2] * u_tnd
        sv_tnd[2:-2, 2:-2] = s[2:-2, 2:-2] * v_tnd

        tendencies = {
            "x_momentum_isentropic": su_tnd,
            "y_momentum_isentropic": sv_tnd,
        }

        return tendencies, {}

    def assert_allclose(self, name, field_a, field_b):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        nb = self.ds.nb
        slc = (slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz))
        try:
            compare_arrays(field_a, field_b, slice=slc)
        except AssertionError:
            raise RuntimeError(f"assert_allclose failed on {name}")


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_smagorinsky(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype, grid_type="numerical", nb_min=2)
    ts = IsentropicSmagorinskyTestSuite(ds)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
