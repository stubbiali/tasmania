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
from sympl import DataArray

import gt4py as gt

from tasmania.python.isentropic.physics.coriolis import IsentropicConservativeCoriolis
from tasmania import get_dataarray_3d

try:
    from .conf import (
        backend as conf_backend,
        default_origin as conf_default_origin,
        nb as conf_nb,
    )
    from .utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_isentropic_state_f,
        st_one_of,
    )
except (ImportError, ModuleNotFoundError):
    from conf import (
        backend as conf_backend,
        default_origin as conf_default_origin,
        nb as conf_nb,
    )
    from utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_isentropic_state_f,
        st_one_of,
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
def test_conservative(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    f = data.draw(st_floats(min_value=0, max_value=1), label="f")

    time = data.draw(hyp_st.datetimes(), label="time")
    backend = data.draw((st_one_of(conf_backend)), label="backend")
    default_origin = data.draw((st_one_of(conf_default_origin)), label="default_origin")
    storage_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            time=time,
            moist=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    nb = nb if grid_type == "numerical" else 0
    x, y = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    coriolis_parameter = DataArray(f, attrs={"units": "rad s^-1"})

    icc = IsentropicConservativeCoriolis(
        domain,
        grid_type,
        coriolis_parameter,
        backend=backend,
        dtype=grid.x.dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )

    assert "x_momentum_isentropic" in icc.input_properties
    assert "y_momentum_isentropic" in icc.input_properties

    assert "x_momentum_isentropic" in icc.tendency_properties
    assert "y_momentum_isentropic" in icc.tendency_properties

    assert icc.diagnostic_properties == {}

    tendencies, diagnostics = icc(state)

    su_val = get_dataarray_3d(
        f * state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values,
        grid,
        "kg m^-1 K^-1 s^-2",
        grid_shape=(grid.nx, grid.ny, grid.nz),
        set_coordinates=False,
    )
    assert "x_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["x_momentum_isentropic"][x, y],
        su_val[x, y],
        compare_coordinate_values=False,
    )

    sv_val = get_dataarray_3d(
        -f * state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values,
        grid,
        "kg m^-1 K^-1 s^-2",
        grid_shape=(grid.nx, grid.ny, grid.nz),
        set_coordinates=False,
    )
    assert "y_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["y_momentum_isentropic"][x, y],
        sv_val[x, y],
        compare_coordinate_values=False,
    )

    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
