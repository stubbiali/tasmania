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
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.physics.horizontal_smoothing import (
    IsentropicHorizontalSmoothing,
)
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.utils.storage_utils import get_storage_descriptor
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_one_of,
        st_isentropic_state_f,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import (
        compare_dataarrays,
        st_domain,
        st_floats,
        st_one_of,
        st_isentropic_state_f,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(2, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    smooth_coeff = data.draw(st_floats(min_value=0, max_value=1))
    smooth_coeff_max = data.draw(st_floats(min_value=smooth_coeff, max_value=1))
    smooth_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))
    smooth_moist_coeff = data.draw(st_floats(min_value=0, max_value=1))
    smooth_moist_coeff_max = data.draw(
        st_floats(min_value=smooth_moist_coeff, max_value=1)
    )
    smooth_moist_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    smooth_types = ("first_order", "second_order", "third_order")

    descriptor = get_storage_descriptor((nx, ny, nz), dtype, halo=halo)
    in_st = gt.storage.zeros(descriptor, backend=backend)
    out_st = gt.storage.zeros(descriptor, backend=backend)

    for smooth_type in smooth_types:
        #
        # validation data
        #
        hs = HorizontalSmoothing.factory(
            smooth_type,
            (nx, ny, nz),
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            backend=backend,
            dtype=dtype,
            halo=halo,
            rebuild=True,
        )
        hs_moist = HorizontalSmoothing.factory(
            smooth_type,
            (nx, ny, nz),
            smooth_moist_coeff,
            smooth_moist_coeff_max,
            smooth_moist_damp_depth,
            nb,
            backend=backend,
            dtype=dtype,
            halo=halo,
            rebuild=False,
        )

        val = {}

        names = (
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
        )
        units = ("kg m^-2 K^-1", "kg m^-1 K^-1 s^-1", "kg m^-1 K^-1 s^-1")
        for i in range(len(names)):
            in_st.data[...] = state[names[i]].to_units(units[i]).values
            hs(in_st, out_st)
            val[names[i]] = deepcopy(out_st.data)

        names = (mfwv, mfcw, mfpw)
        units = ("g g^-1",) * 3
        for i in range(len(names)):
            in_st.data[...] = state[names[i]].to_units(units[i]).values
            hs_moist(in_st, out_st)
            val[names[i]] = deepcopy(out_st.data)

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
            dtype=dtype,
            halo=halo,
        )

        diagnostics = ihs(state)

        names = (
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
        )
        units = ("kg m^-2 K^-1", "kg m^-1 K^-1 s^-1", "kg m^-1 K^-1 s^-1")
        for i in range(len(names)):
            assert names[i] in diagnostics
            field_val = get_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
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
            dtype=dtype,
            halo=halo,
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
            assert names[i] in diagnostics
            field_val = get_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
            compare_dataarrays(diagnostics[names[i]], field_val)

        assert len(diagnostics) == len(names)


if __name__ == "__main__":
    pytest.main([__file__])
