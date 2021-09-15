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

from tasmania.python.isentropic.physics.coriolis import (
    IsentropicConservativeCoriolis,
)

from tests import conf
from tests.strategies import (
    st_floats,
    st_isentropic_state_f,
)
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import hyp_settings


class IsentropicConservativeCoriolisTestSuite(TendencyComponentTestSuite):
    def __init__(self, domain_suite):
        self.f = domain_suite.hyp_data.draw(st_floats(), label="f")
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        f = DataArray(self.f, attrs={"units": "rad s^-1"})
        return IsentropicConservativeCoriolis(
            self.ds.domain,
            self.ds.grid_type,
            f,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        time = self.hyp_data.draw(hyp_st.datetimes(), label="time")
        return self.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                time=time,
                moist=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz
        nb = self.ds.nb if self.ds.grid_type == "numerical" else 0
        i, j, k = slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz)
        su = raw_state_np["x_momentum_isentropic"]
        sv = raw_state_np["y_momentum_isentropic"]
        su_tnd = np.zeros_like(su)
        su_tnd[i, j, k] = self.f * sv[i, j, k]
        sv_tnd = np.zeros_like(su)
        sv_tnd[i, j, k] = -self.f * su[i, j, k]
        tendencies = {
            "x_momentum_isentropic": su_tnd,
            "y_momentum_isentropic": sv_tnd,
        }
        diagnostics = {}
        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_conservative(data, backend, dtype):
    ds = DomainSuite(data, backend, dtype)
    ts = IsentropicConservativeCoriolisTestSuite(ds)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
