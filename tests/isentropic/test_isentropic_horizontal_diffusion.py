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
from sympl import DataArray

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.horizontal_diffusion import (
    IsentropicHorizontalDiffusion,
)
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.utils.storage import get_dataarray_3d

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_default_origin,
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
@pytest.mark.parametrize("diff_type", ("second_order", "fourth_order"))
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, diff_type, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(
        st_one_of(conf_default_origin), label="default_origin"
    )

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
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

    diff_coeff = data.draw(st_floats(min_value=0, max_value=1))
    diff_coeff_max = data.draw(st_floats(min_value=diff_coeff, max_value=1))
    diff_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz)
    )
    diff_moist_coeff = data.draw(st_floats(min_value=0, max_value=1))
    diff_moist_coeff_max = data.draw(
        st_floats(min_value=diff_moist_coeff, max_value=1)
    )
    diff_moist_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz)
    )

    # ========================================
    # test bed
    # ========================================
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    in_st = zeros(backend, shape=storage_shape, storage_options=so)
    out_st = zeros(backend, shape=storage_shape, storage_options=so)

    #
    # validation data
    #
    hd = HorizontalDiffusion.factory(
        diff_type,
        storage_shape,
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    hd_moist = HorizontalDiffusion.factory(
        diff_type,
        storage_shape,
        dx,
        dy,
        diff_moist_coeff,
        diff_moist_coeff_max,
        diff_moist_damp_depth,
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
        hd(in_st, out_st)
        val[names[i]] = deepcopy(out_st)

    names = (mfwv, mfcw, mfpw)
    units = ("g g^-1",) * 3
    for i in range(len(names)):
        in_st[...] = state[names[i]].to_units(units[i]).data
        hd_moist(in_st, out_st)
        val[names[i]] = deepcopy(out_st)

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
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    tendencies, diagnostics = ihd(state)

    names = (
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    )
    units = ("kg m^-2 K^-1 s^-1", "kg m^-1 K^-1 s^-2", "kg m^-1 K^-1 s^-2")
    for i in range(len(names)):
        # with subtests.test(diff_type=diff_type, name=names[i]):
        assert names[i] in tendencies
        field_val = get_dataarray_3d(
            val[names[i]],
            grid,
            units[i],
            name=names[i],
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
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
        diffusion_moist_coeff=DataArray(
            diff_moist_coeff, attrs={"units": "s^-1"}
        ),
        diffusion_moist_coeff_max=DataArray(
            diff_moist_coeff_max, attrs={"units": "s^-1"}
        ),
        diffusion_moist_damp_depth=diff_moist_damp_depth,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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
        # with subtests.test(diff_type=diff_type, name=names[i]):
        assert names[i] in tendencies
        field_val = get_dataarray_3d(
            val[names[i]],
            grid,
            units[i],
            name=names[i],
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        compare_dataarrays(tendencies[names[i]], field_val)

    assert len(tendencies) == len(names)

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
