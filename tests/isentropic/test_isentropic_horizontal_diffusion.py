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

from sympl import DataArray

from tasmania.python.isentropic.physics.horizontal_diffusion import (
    IsentropicHorizontalDiffusion,
)
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion

from tests import conf
from tests.strategies import (
    st_floats,
    st_isentropic_state_f,
)
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicHorizontalDiffusionTestSuite(TendencyComponentTestSuite):
    def __init__(self, domain_suite, diff_type, moist):
        self.diff_type = diff_type
        self.moist = moist

        hyp_data = domain_suite.hyp_data
        self.diff_coeff = hyp_data.draw(st_floats(min_value=0, max_value=1))
        self.diff_coeff_max = hyp_data.draw(
            st_floats(min_value=self.diff_coeff, max_value=1)
        )
        self.diff_damp_depth = hyp_data.draw(
            hyp_st.integers(min_value=0, max_value=domain_suite.grid.nz)
        )
        self.diff_moist_coeff = hyp_data.draw(
            st_floats(min_value=0, max_value=1)
        )
        self.diff_moist_coeff_max = hyp_data.draw(
            st_floats(min_value=self.diff_moist_coeff, max_value=1)
        )
        self.diff_moist_damp_depth = hyp_data.draw(
            hyp_st.integers(min_value=0, max_value=domain_suite.grid.nz)
        )

        super().__init__(domain_suite)

    @cached_property
    def component(self):
        if not self.moist:
            return IsentropicHorizontalDiffusion(
                self.ds.domain,
                self.diff_type,
                diffusion_coeff=DataArray(
                    self.diff_coeff, attrs={"units": "s^-1"}
                ),
                diffusion_coeff_max=DataArray(
                    self.diff_coeff_max, attrs={"units": "s^-1"}
                ),
                diffusion_damp_depth=self.diff_damp_depth,
                backend=self.ds.backend,
                backend_options=self.ds.bo,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )
        else:
            return IsentropicHorizontalDiffusion(
                self.ds.domain,
                self.diff_type,
                diffusion_coeff=DataArray(
                    self.diff_coeff, attrs={"units": "s^-1"}
                ),
                diffusion_coeff_max=DataArray(
                    self.diff_coeff_max, attrs={"units": "s^-1"}
                ),
                diffusion_damp_depth=self.diff_damp_depth,
                moist=True,
                diffusion_moist_coeff=DataArray(
                    self.diff_moist_coeff, attrs={"units": "s^-1"}
                ),
                diffusion_moist_coeff_max=DataArray(
                    self.diff_moist_coeff_max, attrs={"units": "s^-1"}
                ),
                diffusion_moist_damp_depth=self.diff_moist_damp_depth,
                backend=self.ds.backend,
                backend_options=self.ds.bo,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )

    def get_state(self):
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=self.moist,
                precipitation=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        dx = self.ds.grid.dx.to_units("m").values.item()
        dy = self.ds.grid.dy.to_units("m").values.item()

        hd = HorizontalDiffusion.factory(
            self.diff_type,
            self.ds.storage_shape or (nx, ny, nz),
            dx,
            dy,
            self.diff_coeff,
            self.diff_coeff_max,
            self.diff_damp_depth,
            self.ds.nb,
            backend="numpy",
            backend_options=self.ds.bo,
            storage_options=self.ds.so,
        )
        s = raw_state_np["air_isentropic_density"]
        s_tnd = np.zeros_like(s)
        hd(s, s_tnd, overwrite_output=True)
        su = raw_state_np["x_momentum_isentropic"]
        su_tnd = np.zeros_like(su)
        hd(su, su_tnd, overwrite_output=True)
        sv = raw_state_np["y_momentum_isentropic"]
        sv_tnd = np.zeros_like(sv)
        hd(sv, sv_tnd, overwrite_output=True)
        tendencies = {
            "air_isentropic_density": s_tnd,
            "x_momentum_isentropic": su_tnd,
            "y_momentum_isentropic": sv_tnd,
        }

        if self.moist:
            hd_moist = HorizontalDiffusion.factory(
                self.diff_type,
                self.ds.storage_shape or (nx, ny, nz),
                dx,
                dy,
                self.diff_moist_coeff,
                self.diff_moist_coeff_max,
                self.diff_moist_damp_depth,
                self.ds.nb,
                backend="numpy",
                backend_options=self.ds.bo,
                storage_options=self.ds.so,
            )
            qv = raw_state_np[mfwv]
            qv_tnd = np.zeros_like(qv)
            hd_moist(qv, qv_tnd, overwrite_output=True)
            tendencies[mfwv] = qv_tnd
            qc = raw_state_np[mfcw]
            qc_tnd = np.zeros_like(qc)
            hd_moist(qc, qc_tnd, overwrite_output=True)
            tendencies[mfcw] = qc_tnd
            qr = raw_state_np[mfpw]
            qr_tnd = np.zeros_like(qr)
            hd_moist(qr, qr_tnd, overwrite_output=True)
            tendencies[mfpw] = qr_tnd

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
@pytest.mark.parametrize("diff_type", ("second_order", "fourth_order"))
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, diff_type, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype, grid_type="numerical", nb_min=2)

    # ========================================
    # test bed
    # ========================================
    # dry
    ts = IsentropicHorizontalDiffusionTestSuite(ds, diff_type, False)
    ts.run()

    # moist
    ts = IsentropicHorizontalDiffusionTestSuite(ds, diff_type, True)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
