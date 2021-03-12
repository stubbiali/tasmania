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
from sympl import DataArray

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.coriolis import (
    IsentropicConservativeCoriolis,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import (
    st_domain,
    st_floats,
    st_isentropic_state_f,
    st_one_of,
)
from tests.utilities import compare_dataarrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_conservative(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        (st_one_of(conf.aligned_index)), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            nb=nb, backend=backend, backend_options=bo, storage_options=so
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    f = data.draw(st_floats(min_value=0, max_value=1), label="f")

    time = data.draw(hyp_st.datetimes(), label="time")

    state = data.draw(
        st_isentropic_state_f(
            grid,
            time=time,
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
    nb = nb if grid_type == "numerical" else 0
    x, y, z = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb), slice(0, nz)
    coriolis_parameter = DataArray(f, attrs={"units": "rad s^-1"})

    icc = IsentropicConservativeCoriolis(
        domain,
        grid_type,
        coriolis_parameter,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    assert "x_momentum_isentropic" in icc.input_properties
    assert "y_momentum_isentropic" in icc.input_properties

    assert "x_momentum_isentropic" in icc.tendency_properties
    assert "y_momentum_isentropic" in icc.tendency_properties

    assert icc.diagnostic_properties == {}

    tendencies, diagnostics = icc(state)

    su_val_np = f * to_numpy(
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    su_val = get_dataarray_3d(
        su_val_np,
        grid,
        "kg m^-1 K^-1 s^-2",
        grid_shape=(nx, ny, nz),
        set_coordinates=False,
    )
    assert "x_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["x_momentum_isentropic"],
        su_val,
        compare_coordinate_values=False,
        slice=(x, y, z),
    )

    sv_val_np = -f * to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv_val = get_dataarray_3d(
        sv_val_np,
        grid,
        "kg m^-1 K^-1 s^-2",
        grid_shape=(nx, ny, nz),
        set_coordinates=False,
    )
    assert "y_momentum_isentropic" in tendencies
    compare_dataarrays(
        tendencies["y_momentum_isentropic"],
        sv_val,
        compare_coordinate_values=False,
        slice=(x, y, z),
    )

    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
