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
from copy import deepcopy
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.horizontal_smoothing import (
    IsentropicHorizontalSmoothing,
)
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.utils.storage import get_dataarray_3d

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import (
    st_domain,
    st_floats,
    st_one_of,
    st_isentropic_state_f,
)
from tests.utilities import compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "smooth_type", ("first_order", "second_order", "third_order")
)
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, smooth_type, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    smooth_coeff = data.draw(st_floats(min_value=0, max_value=1))
    smooth_coeff_max = data.draw(
        st_floats(min_value=smooth_coeff, max_value=1)
    )
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz)
    )
    smooth_moist_coeff = data.draw(st_floats(min_value=0, max_value=1))
    smooth_moist_coeff_max = data.draw(
        st_floats(min_value=smooth_moist_coeff, max_value=1)
    )
    smooth_moist_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz)
    )

    # ========================================
    # test bed
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    in_st = zeros(backend, shape=storage_shape, storage_options=so)
    out_st = zeros(backend, shape=storage_shape, storage_options=so)

    #
    # validation data
    #
    hs = HorizontalSmoothing.factory(
        smooth_type,
        storage_shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    hs_moist = HorizontalSmoothing.factory(
        smooth_type,
        storage_shape,
        smooth_moist_coeff,
        smooth_moist_coeff_max,
        smooth_moist_damp_depth,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    val = {}

    names = (
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    )
    units = ("kg m^-2 K^-1", "kg m^-1 K^-1 s^-1", "kg m^-1 K^-1 s^-1")
    for i in range(len(names)):
        in_st[...] = state[names[i]].to_units(units[i]).data
        hs(in_st, out_st)
        val[names[i]] = deepcopy(out_st)

    names = (mfwv, mfcw, mfpw)
    units = ("g g^-1",) * 3
    for i in range(len(names)):
        in_st[...] = state[names[i]].to_units(units[i]).data
        hs_moist(in_st, out_st)
        val[names[i]] = deepcopy(out_st)

    #
    # dry
    #
    ihs = IsentropicHorizontalSmoothing(
        domain,
        smooth_type,
        smooth_coeff=smooth_coeff,
        smooth_coeff_max=smooth_coeff_max,
        smooth_damp_depth=smooth_damp_depth,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    diagnostics = ihs(state)

    names = (
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    )
    units = ("kg m^-2 K^-1", "kg m^-1 K^-1 s^-1", "kg m^-1 K^-1 s^-1")
    for i in range(len(names)):
        # with subtests.test(smooth_type=smooth_type, name=names[i]):
        assert names[i] in diagnostics
        field_val = get_dataarray_3d(
            val[names[i]],
            grid,
            units[i],
            name=names[i],
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        compare_dataarrays(diagnostics[names[i]], field_val)

    assert len(diagnostics) == len(names)

    #
    # moist
    #
    ihs = IsentropicHorizontalSmoothing(
        domain,
        smooth_type,
        smooth_coeff=smooth_coeff,
        smooth_coeff_max=smooth_coeff_max,
        smooth_damp_depth=smooth_damp_depth,
        moist=True,
        smooth_moist_coeff=smooth_moist_coeff,
        smooth_moist_coeff_max=smooth_moist_coeff_max,
        smooth_moist_damp_depth=smooth_moist_damp_depth,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    diagnostics = ihs(state)

    names = (
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        mfwv,
        mfcw,
        mfpw,
    )
    units = (
        "kg m^-2 K^-1",
        "kg m^-1 K^-1 s^-1",
        "kg m^-1 K^-1 s^-1",
        "g g^-1",
        "g g^-1",
        "g g^-1",
    )
    for i in range(len(names)):
        # with subtests.test(smooth_type=smooth_type, name=names[i]):
        assert names[i] in diagnostics
        field_val = get_dataarray_3d(
            val[names[i]],
            grid,
            units[i],
            name=names[i],
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        compare_dataarrays(diagnostics[names[i]], field_val)

    assert len(diagnostics) == len(names)


if __name__ == "__main__":
    pytest.main([__file__])
