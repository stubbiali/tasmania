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
import pytest

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.turbulence import IsentropicSmagorinsky
from tasmania import get_dataarray_3d

from tests import conf
from tests.physics.test_turbulence import smagorinsky2d_validation
from tests.strategies import st_domain, st_one_of, st_isentropic_state_f
from tests.utilities import compare_dataarrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_smagorinsky(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    cs = data.draw(hyp_st.floats(min_value=0, max_value=10), label="cs")

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv = to_numpy(
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )

    u = su / s
    v = sv / s
    u_tnd, v_tnd = smagorinsky2d_validation(dx, dy, cs, u, v)

    smag = IsentropicSmagorinsky(
        domain,
        smagorinsky_constant=cs,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    tendencies, diagnostics = smag(state)

    slc = (slice(nb, nx - nb), slice(nb, ny - nb), slice(nz))

    assert "x_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["x_momentum_isentropic"],
        get_dataarray_3d(
            s * u_tnd,
            grid,
            "kg m^-1 K^-1 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=slc,
    )
    assert "y_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["y_momentum_isentropic"],
        get_dataarray_3d(
            s * v_tnd,
            grid,
            "kg m^-1 K^-1 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=slc,
    )
    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
