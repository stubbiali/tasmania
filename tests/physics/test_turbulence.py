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

from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_raw_field
from tests.suites import DomainSuite, TendencyComponentTestSuite
from tests.utilities import compare_arrays, hyp_settings


class Smagorinsky2dTestSuite(TendencyComponentTestSuite):
    def __init__(self, domain_suite):
        super().__init__(domain_suite)
        self.storage_shape = (
            self.storage_shape
            if self.storage_shape
            else (self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz)
        )

    @cached_property
    def component(self):
        cs = self.hyp_data.draw(
            hyp_st.floats(min_value=0, max_value=10), label="cs"
        )
        return Smagorinsky2d(
            self.ds.domain,
            smagorinsky_constant=cs,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        time = self.hyp_data.draw(hyp_st.datetimes(), label="time")
        u = self.hyp_data.draw(
            st_raw_field(
                self.storage_shape,
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="u",
        )
        v = self.hyp_data.draw(
            st_raw_field(
                self.storage_shape,
                -1e3,
                1e3,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="v",
        )

        g = self.ds.grid
        state = {
            "time": time,
            "x_velocity": get_dataarray_3d(
                u,
                g,
                "m s^-1",
                grid_shape=(g.nx, g.ny, g.nz),
                set_coordinates=False,
            ),
            "y_velocity": get_dataarray_3d(
                v,
                g,
                "m s^-1",
                grid_shape=(g.nx, g.ny, g.nz),
                set_coordinates=False,
            ),
        }

        return state

    def get_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        cs = self.component._cs
        dx = self.ds.grid.dx.to_units("m").values.item()
        dy = self.ds.grid.dy.to_units("m").values.item()
        u = raw_state_np["x_velocity"]
        v = raw_state_np["y_velocity"]

        u_tnd = np.zeros_like(u)
        v_tnd = np.zeros_like(v)

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
        u_tnd[2:-2, 2:-2] = 2.0 * (
            (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2])
            / (2.0 * dy)
        )
        v_tnd[2:-2, 2:-2] = 2.0 * (
            (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1])
            / (2.0 * dx)
            + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2])
            / (2.0 * dy)
        )

        tendencies = {"x_velocity": u_tnd, "y_velocity": v_tnd}

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
def test_smagorinsky2d(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype, grid_type="numerical", nb_min=2)
    ts = Smagorinsky2dTestSuite(ds)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
