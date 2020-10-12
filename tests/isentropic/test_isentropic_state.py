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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest
from sympl import DataArray

from tasmania.python.isentropic.state import (
    get_isentropic_state_from_brunt_vaisala_frequency,
)

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import st_one_of, st_domain
from tests.utilities import hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_brunt_vaisala(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20),
            yaxis_length=(1, 20),
            zaxis_length=(2, 10),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    # ========================================
    # test bed
    # ========================================
    state = get_isentropic_state_from_brunt_vaisala_frequency(
        grid=grid,
        time=datetime(year=1992, month=2, day=20, hour=12),
        x_velocity=DataArray(3.4, attrs={"units": "m s^-1"}),
        y_velocity=DataArray(-7.1, attrs={"units": "km hr^-1"}),
        brunt_vaisala=DataArray(0.01, attrs={"units": "s^-1"}),
        moist=False,
        precipitation=False,
        relative_humidity=0.5,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
        managed_memory=False,
    )


if __name__ == "__main__":
    pytest.main([__file__])
