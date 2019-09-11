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
from sympl import DataArray

import gridtools as gt
from tasmania.python.isentropic.physics.horizontal_diffusion import (
    IsentropicHorizontalDiffusion,
)
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.storage_utils import get_storage_descriptor

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
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(2, 20), nb=nb
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    diff_coeff = data.draw(st_floats(min_value=0, max_value=1))
    diff_coeff_max = data.draw(st_floats(min_value=diff_coeff, max_value=1))
    diff_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))
    diff_moist_coeff = data.draw(st_floats(min_value=0, max_value=1))
    diff_moist_coeff_max = data.draw(st_floats(min_value=diff_moist_coeff, max_value=1))
    diff_moist_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    diff_types = ("second_order", "fourth_order")

    descriptor = get_storage_descriptor((nx, ny, nz), dtype, halo=halo)
    in_st = gt.storage.zeros(descriptor, backend=backend)
    out_st = gt.storage.zeros(descriptor, backend=backend)

    for diff_type in diff_types:
        #
        # validation data
        #
        hd = HorizontalDiffusion.factory(
            diff_type,
            (nx, ny, nz),
            dx,
            dy,
            diff_coeff,
            diff_coeff_max,
            diff_damp_depth,
            nb,
            backend=backend,
            dtype=dtype,
            halo=halo,
            rebuild=True,
        )
        hd_moist = HorizontalDiffusion.factory(
            diff_type,
            (nx, ny, nz),
            dx,
            dy,
            diff_moist_coeff,
            diff_moist_coeff_max,
            diff_moist_damp_depth,
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
            hd(in_st, out_st)
            val[names[i]] = deepcopy(out_st.data)

        names = (mfwv, mfcw, mfpw)
        units = ("g g^-1",) * 3
        for i in range(len(names)):
            in_st.data[...] = state[names[i]].to_units(units[i]).values
            hd_moist(in_st, out_st)
            val[names[i]] = deepcopy(out_st.data)

        #
        # dry
        #
        ihd = IsentropicHorizontalDiffusion(
            domain,
            diff_type,
            diffusion_coeff=DataArray(diff_coeff, attrs={"units": "s^-1"}),
            diffusion_coeff_max=DataArray(diff_coeff_max, attrs={"units": "s^-1"}),
            diffusion_damp_depth=diff_damp_depth,
            backend=backend,
            dtype=dtype,
            halo=halo,
        )

        tendencies, diagnostics = ihd(state)

        names = (
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
        )
        units = ("kg m^-2 K^-1 s^-1", "kg m^-1 K^-1 s^-2", "kg m^-1 K^-1 s^-2")
        for i in range(len(names)):
            assert names[i] in tendencies
            field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
            compare_dataarrays(tendencies[names[i]], field_val)

        assert len(tendencies) == len(names)

        assert len(diagnostics) == 0

        #
        # moist
        #
        ihd = IsentropicHorizontalDiffusion(
            domain,
            diff_type,
            diffusion_coeff=DataArray(diff_coeff, attrs={"units": "s^-1"}),
            diffusion_coeff_max=DataArray(diff_coeff_max, attrs={"units": "s^-1"}),
            diffusion_damp_depth=diff_damp_depth,
            moist=True,
            diffusion_moist_coeff=DataArray(diff_moist_coeff, attrs={"units": "s^-1"}),
            diffusion_moist_coeff_max=DataArray(
                diff_moist_coeff_max, attrs={"units": "s^-1"}
            ),
            diffusion_moist_damp_depth=diff_moist_damp_depth,
            backend=backend,
            dtype=dtype,
            halo=halo,
        )

        tendencies, diagnostics = ihd(state)

        names = (
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
            mfwv,
            mfcw,
            mfpw,
        )
        units = (
            "kg m^-2 K^-1 s^-1",
            "kg m^-1 K^-1 s^-2",
            "kg m^-1 K^-1 s^-2",
            "g g^-1 s^-1",
            "g g^-1 s^-1",
            "g g^-1 s^-1",
        )
        for i in range(len(names)):
            assert names[i] in tendencies
            field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
            compare_dataarrays(tendencies[names[i]], field_val)

        assert len(tendencies) == len(names)

        assert len(diagnostics) == 0


if __name__ == "__main__":
    # pytest.main([__file__])
    test()
