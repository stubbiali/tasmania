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
    reproduce_failure,
    strategies as hyp_st,
)
import pytest
from sympl import DataArray

from tasmania.python.burgers.physics.diffusion import (
    BurgersHorizontalDiffusion,
)
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests import conf
from tests.dwarfs.horizontal_diffusers.test_fourth_order import (
    fourth_order_diffusion_xyz,
    fourth_order_diffusion_xz,
    fourth_order_diffusion_yz,
    assert_xyz,
    assert_xz,
    assert_yz,
)
from tests.dwarfs.horizontal_diffusers.test_second_order import (
    second_order_diffusion_xyz,
    second_order_diffusion_xz,
    second_order_diffusion_yz,
)
from tests.strategies import st_burgers_state, st_domain, st_floats, st_one_of
from tests.utilities import hyp_settings


def second_order_validation(grid, smooth_coeff, phi, phi_tnd, nb):
    nx, ny = grid.nx, grid.ny
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    if nx < 3:
        phi_tnd_assert = smooth_coeff * second_order_diffusion_yz(dy, phi)
        assert_yz(phi_tnd, phi_tnd_assert, nb)
    elif ny < 3:
        phi_tnd_assert = smooth_coeff * second_order_diffusion_xz(dx, phi)
        assert_xz(phi_tnd, phi_tnd_assert, nb)
    else:
        phi_tnd_assert = smooth_coeff * second_order_diffusion_xyz(dx, dy, phi)
        assert_xyz(phi_tnd, phi_tnd_assert, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_second_order(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    pgrid = domain.physical_grid
    assume(pgrid.nx > 2 or pgrid.ny > 2)
    ngrid = domain.numerical_grid

    pstate = data.draw(
        st_burgers_state(pgrid, backend=backend, storage_options=so),
        label="pstate",
    )
    nstate = data.draw(
        st_burgers_state(ngrid, backend=backend, storage_options=so),
        label="nstate",
    )

    smooth_coeff = data.draw(
        st_floats(min_value=0, max_value=1), label="smooth_coeff"
    )

    # ========================================
    # test
    # ========================================
    #
    # physical grid
    #
    order = "second_order"
    if pgrid.nx < 3:
        order += "_1dy"
    elif pgrid.ny < 3:
        order += "_1dx"

    pbhd = BurgersHorizontalDiffusion(
        domain,
        "physical",
        order,
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    tendencies, diagnostics = pbhd(pstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        pgrid,
        smooth_coeff,
        to_numpy(pstate["x_velocity"].to_units("m s^-1").data),
        tendencies["x_velocity"].data,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        pgrid,
        smooth_coeff,
        to_numpy(pstate["y_velocity"].to_units("m s^-1").data),
        tendencies["y_velocity"].data,
        nb,
    )

    #
    # numerical grid
    #
    cbhd = BurgersHorizontalDiffusion(
        domain,
        "numerical",
        "second_order",
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    tendencies, diagnostics = cbhd(nstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        ngrid,
        smooth_coeff,
        to_numpy(nstate["x_velocity"].to_units("m s^-1").data),
        tendencies["x_velocity"].data,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        ngrid,
        smooth_coeff,
        to_numpy(nstate["y_velocity"].to_units("m s^-1").data),
        tendencies["y_velocity"].data,
        nb,
    )


def fourth_order_validation(grid, smooth_coeff, phi, phi_tnd, nb):
    nx, ny = grid.nx, grid.ny
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    if nx < 5:
        phi_tnd_assert = smooth_coeff * fourth_order_diffusion_yz(dy, phi)
        assert_yz(phi_tnd, phi_tnd_assert, nb)
    elif ny < 5:
        phi_tnd_assert = smooth_coeff * fourth_order_diffusion_xz(dx, phi)
        assert_xz(phi_tnd, phi_tnd_assert, nb)
    else:
        phi_tnd_assert = smooth_coeff * fourth_order_diffusion_xyz(dx, dy, phi)
        assert_xyz(phi_tnd, phi_tnd_assert, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_fourth_order(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf.nb)))
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    pgrid = domain.physical_grid
    assume(pgrid.nx > 4 or pgrid.ny > 4)
    ngrid = domain.numerical_grid

    pstate = data.draw(
        st_burgers_state(pgrid, backend=backend, storage_options=so),
        label="pstate",
    )
    nstate = data.draw(
        st_burgers_state(ngrid, backend=backend, storage_options=so),
        label="nstate",
    )

    smooth_coeff = data.draw(
        st_floats(min_value=0, max_value=1), label="smooth_coeff"
    )

    # ========================================
    # test
    # ========================================
    #
    # physical grid
    #
    order = "fourth_order"
    if pgrid.nx < 4:
        order += "_1dy"
    elif pgrid.ny < 4:
        order += "_1dx"

    pbhd = BurgersHorizontalDiffusion(
        domain,
        "physical",
        order,
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    tendencies, diagnostics = pbhd(pstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        pgrid,
        smooth_coeff,
        to_numpy(pstate["x_velocity"].to_units("m s^-1").data),
        tendencies["x_velocity"].data,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        pgrid,
        smooth_coeff,
        to_numpy(pstate["y_velocity"].to_units("m s^-1").data),
        tendencies["y_velocity"].data,
        nb,
    )

    #
    # numerical grid
    #
    cbhd = BurgersHorizontalDiffusion(
        domain,
        "numerical",
        "fourth_order",
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    tendencies, diagnostics = cbhd(nstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        ngrid,
        smooth_coeff,
        to_numpy(nstate["x_velocity"].to_units("m s^-1").data),
        tendencies["x_velocity"].data,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        ngrid,
        smooth_coeff,
        to_numpy(nstate["y_velocity"].to_units("m s^-1").data),
        tendencies["y_velocity"].data,
        nb,
    )


if __name__ == "__main__":
    pytest.main([__file__])
