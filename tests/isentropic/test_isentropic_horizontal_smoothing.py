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

from tasmania.python.isentropic.physics.horizontal_smoothing import (
    IsentropicHorizontalSmoothing,
)
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing

from tests import conf
from tests.strategies import (
    st_floats,
    st_isentropic_state_f,
)
from tests.suites import DiagnosticComponentTestSuite, DomainSuite
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicHorizontalSmoothingTestSuite(DiagnosticComponentTestSuite):
    def __init__(
        self, hyp_data, domain_suite, smooth_type, moist, *, storage_shape
    ):
        self.smooth_type = smooth_type
        self.moist = moist
        self.storage_shape = storage_shape

        self.smooth_coeff = hyp_data.draw(st_floats(min_value=0, max_value=1))
        self.smooth_coeff_max = hyp_data.draw(
            st_floats(min_value=self.smooth_coeff, max_value=1)
        )
        self.smooth_damp_depth = hyp_data.draw(
            hyp_st.integers(min_value=0, max_value=domain_suite.grid.nz)
        )
        self.smooth_moist_coeff = hyp_data.draw(
            st_floats(min_value=0, max_value=1)
        )
        self.smooth_moist_coeff_max = hyp_data.draw(
            st_floats(min_value=self.smooth_moist_coeff, max_value=1)
        )
        self.smooth_moist_damp_depth = hyp_data.draw(
            hyp_st.integers(min_value=0, max_value=domain_suite.grid.nz)
        )

        super().__init__(hyp_data, domain_suite)

    @cached_property
    def component(self):
        if not self.moist:
            return IsentropicHorizontalSmoothing(
                self.ds.domain,
                self.smooth_type,
                smooth_coeff=self.smooth_coeff,
                smooth_coeff_max=self.smooth_coeff_max,
                smooth_damp_depth=self.smooth_damp_depth,
                moist=False,
                backend=self.ds.backend,
                backend_options=self.ds.bo,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            )
        else:
            return IsentropicHorizontalSmoothing(
                self.ds.domain,
                self.smooth_type,
                smooth_coeff=self.smooth_coeff,
                smooth_coeff_max=self.smooth_coeff_max,
                smooth_damp_depth=self.smooth_damp_depth,
                moist=True,
                smooth_moist_coeff=self.smooth_moist_coeff,
                smooth_moist_coeff_max=self.smooth_moist_coeff_max,
                smooth_moist_damp_depth=self.smooth_moist_damp_depth,
                backend=self.ds.backend,
                backend_options=self.ds.bo,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            )

    def get_state(self):
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=self.moist,
                backend=self.ds.backend,
                storage_shape=self.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_diagnostics(self, raw_state_np):
        hs = HorizontalSmoothing.factory(
            self.smooth_type,
            self.storage_shape,
            self.smooth_coeff,
            self.smooth_coeff_max,
            self.smooth_damp_depth,
            self.ds.nb,
            backend="numpy",
            backend_options=self.ds.bo,
            storage_options=self.ds.so,
        )
        s = raw_state_np["air_isentropic_density"]
        s_out = np.zeros_like(s)
        hs(s, s_out)
        su = raw_state_np["x_momentum_isentropic"]
        su_out = np.zeros_like(su)
        hs(su, su_out)
        sv = raw_state_np["y_momentum_isentropic"]
        sv_out = np.zeros_like(sv)
        hs(sv, sv_out)
        diagnostics = {
            "air_isentropic_density": s_out,
            "x_momentum_isentropic": su_out,
            "y_momentum_isentropic": sv_out,
        }

        if self.moist:
            hs_moist = HorizontalSmoothing.factory(
                self.smooth_type,
                self.storage_shape,
                self.smooth_moist_coeff,
                self.smooth_moist_coeff_max,
                self.smooth_moist_damp_depth,
                self.ds.nb,
                backend="numpy",
                backend_options=self.ds.bo,
                storage_options=self.ds.so,
            )
            qv = raw_state_np[mfwv]
            qv_out = np.zeros_like(qv)
            hs_moist(qv, qv_out)
            diagnostics[mfwv] = qv_out
            qc = raw_state_np[mfcw]
            qc_out = np.zeros_like(qc)
            hs_moist(qc, qc_out)
            diagnostics[mfcw] = qc_out
            qr = raw_state_np[mfpw]
            qr_out = np.zeros_like(qr)
            hs_moist(qr, qr_out)
            diagnostics[mfpw] = qr_out

        return diagnostics

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
@pytest.mark.parametrize(
    "smooth_type", ("first_order", "second_order")  # , "third_order")
)
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, smooth_type, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype, grid_type="numerical", nb_min=3)
    storage_shape = (ds.grid.nx + 1, ds.grid.ny + 1, ds.grid.nz + 1)

    # ========================================
    # test bed
    # ========================================
    # dry
    ts = IsentropicHorizontalSmoothingTestSuite(
        data, ds, smooth_type, False, storage_shape=storage_shape
    )
    ts.run()

    # moist
    ts = IsentropicHorizontalSmoothingTestSuite(
        data, ds, smooth_type, True, storage_shape=storage_shape
    )
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
