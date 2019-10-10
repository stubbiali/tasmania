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
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import pytest

from tasmania.python.isentropic.physics.turbulence import IsentropicSmagorinsky
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .test_turbulence import smagorinsky2d_validation
    from .utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_one_of,
        st_isentropic_state_f,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from test_turbulence import smagorinsky2d_validation
    from utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_one_of,
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
def test_smagorinsky(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")

    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid

    cs = data.draw(hyp_st.floats(min_value=0, max_value=10), label="cs")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=False, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dtype = grid.x.dtype

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv = state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values

    u = su / s
    v = sv / s
    u_tnd, v_tnd = smagorinsky2d_validation(dx, dy, cs, u, v)

    smag = IsentropicSmagorinsky(
        domain,
        smagorinsky_constant=cs,
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )

    tendencies, diagnostics = smag(state)

    assert "x_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["x_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
        get_dataarray_3d(
            s * u_tnd,
            grid,
            "kg m^-1 K^-1 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )[nb : -nb - 1, nb : -nb - 1, :-1],
        compare_coordinate_values=False,
    )
    assert "y_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["y_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
        get_dataarray_3d(
            s * v_tnd,
            grid,
            "kg m^-1 K^-1 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
            )[nb : -nb - 1, nb : -nb - 1, :-1],
        compare_coordinate_values=False,
    )
    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
