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
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
    Verbosity,
)
import pytest
from sympl import DataArray

import gt4py as gt

from tasmania.python.isentropic.state import (
    get_isentropic_state_from_brunt_vaisala_frequency,
)

from tests.conf import (
    backend as conf_backend,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utilities import (
    compare_arrays,
    compare_datetimes,
    st_floats,
    st_one_of,
    st_domain,
    st_isentropic_state_f,
)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_brunt_vaisala(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(2, 10), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
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
