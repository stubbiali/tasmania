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
    seed,
    settings,
    strategies as hyp_st,
)
import pytest
from sympl import DataArray

from tasmania.python.burgers.physics.diffusion import BurgersHorizontalDiffusion

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .dwarfs.test_horizontal_diffusion import (
        second_order_diffusion_xyz,
        second_order_diffusion_xz,
        second_order_diffusion_yz,
        fourth_order_diffusion_xyz,
        fourth_order_diffusion_xz,
        fourth_order_diffusion_yz,
        assert_xyz,
        assert_xz,
        assert_yz,
    )
    from .utils import st_burgers_state, st_domain, st_floats, st_one_of
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from dwarfs.test_horizontal_diffusion import (
        second_order_diffusion_xyz,
        second_order_diffusion_xz,
        second_order_diffusion_yz,
        fourth_order_diffusion_xyz,
        fourth_order_diffusion_xz,
        fourth_order_diffusion_yz,
        assert_xyz,
        assert_xz,
        assert_yz,
    )
    from utils import st_burgers_state, st_domain, st_floats, st_one_of


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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_second_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(xaxis_length=(1, 40), yaxis_length=(1, 40), zaxis_length=(1, 1), nb=nb),
        label="domain",
    )
    pgrid = domain.physical_grid
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = pgrid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")

    pstate = data.draw(
        st_burgers_state(pgrid, backend=backend, halo=halo), label="pstate"
    )
    cstate = data.draw(
        st_burgers_state(cgrid, backend=backend, halo=halo), label="cstate"
    )

    smooth_coeff = data.draw(st_floats(min_value=0, max_value=1), label="smooth_coeff")

    # ========================================
    # test
    # ========================================
    #
    # physical grid
    #
    pbhd = BurgersHorizontalDiffusion(
        domain,
        "physical",
        "second_order",
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        dtype=pgrid.x.dtype,
        halo=halo,
        rebuild=True,
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
        pstate["x_velocity"].to_units("m s^-1").values,
        tendencies["x_velocity"].values,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        pgrid,
        smooth_coeff,
        pstate["y_velocity"].to_units("m s^-1").values,
        tendencies["y_velocity"].values,
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
        dtype=cgrid.x.dtype,
        halo=halo,
        rebuild=True,
    )

    tendencies, diagnostics = cbhd(cstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        cgrid,
        smooth_coeff,
        cstate["x_velocity"].to_units("m s^-1").values,
        tendencies["x_velocity"].values,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    second_order_validation(
        cgrid,
        smooth_coeff,
        cstate["y_velocity"].to_units("m s^-1").values,
        tendencies["y_velocity"].values,
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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_fourth_order(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)))
    domain = data.draw(
        st_domain(xaxis_length=(1, 40), yaxis_length=(1, 40), zaxis_length=(1, 1), nb=nb),
        label="domain",
    )
    pgrid = domain.physical_grid
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = pgrid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")

    pstate = data.draw(
        st_burgers_state(pgrid, backend=backend, halo=halo), label="pstate"
    )
    cstate = data.draw(
        st_burgers_state(cgrid, backend=backend, halo=halo), label="cstate"
    )

    smooth_coeff = data.draw(st_floats(min_value=0, max_value=1), label="smooth_coeff")

    # ========================================
    # test
    # ========================================
    #
    # physical grid
    #
    pbhd = BurgersHorizontalDiffusion(
        domain,
        "physical",
        "fourth_order",
        DataArray(smooth_coeff, attrs={"units": "m^2 s^-1"}),
        backend=backend,
        dtype=pgrid.x.dtype,
        halo=halo,
        rebuild=True,
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
        pstate["x_velocity"].to_units("m s^-1").values,
        tendencies["x_velocity"].values,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        pgrid,
        smooth_coeff,
        pstate["y_velocity"].to_units("m s^-1").values,
        tendencies["y_velocity"].values,
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
        dtype=cgrid.x.dtype,
        halo=halo,
        rebuild=True,
    )

    tendencies, diagnostics = cbhd(cstate)

    assert len(diagnostics) == 0

    assert "x_velocity" in tendencies
    assert "y_velocity" in tendencies
    assert len(tendencies) == 2

    assert tendencies["x_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        cgrid,
        smooth_coeff,
        cstate["x_velocity"].to_units("m s^-1").values,
        tendencies["x_velocity"].values,
        nb,
    )

    assert tendencies["y_velocity"].attrs["units"] == "m s^-2"
    fourth_order_validation(
        cgrid,
        smooth_coeff,
        cstate["y_velocity"].to_units("m s^-1").values,
        tendencies["y_velocity"].values,
        nb,
    )


if __name__ == "__main__":
    pytest.main([__file__])
