# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from property_cached import cached_property
import pytest

from sympl import DataArray

from tasmania.python.burgers.physics.diffusion import (
    BurgersHorizontalDiffusion,
)

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
from tests.strategies import st_burgers_state, st_floats
from tests.suites.core_components import TendencyComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import hyp_settings


class SecondOrder(TendencyComponentTestSuite):
    def __init__(self, domain_suite):
        self.smooth_coeff = domain_suite.hyp_data.draw(
            st_floats(min_value=0, max_value=1), label="smooth_coeff"
        )
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        order = "second_order"
        if self.ds.grid.nx < 3:
            order += "_1dy"
        elif self.ds.grid.ny < 3:
            order += "_1dx"

        return BurgersHorizontalDiffusion(
            self.ds.domain,
            self.ds.grid_type,
            order,
            DataArray(self.smooth_coeff, attrs={"units": "m^2 s^-1"}),
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_options=self.ds.so,
        )

    def assert_allclose(self, name, field_a, field_b):
        if self.ds.grid.nx < 3:
            assert_yz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)
        elif self.ds.grid.ny < 3:
            assert_xz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)
        else:
            assert_xyz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)

    def get_state(self):
        return self.hyp_data.draw(
            st_burgers_state(
                self.ds.grid,
                backend=self.ds.backend,
                storage_options=self.ds.so,
            ),
            label="state",
        )

    def get_validation_tendencies_and_diagnostics(self, raw_state_np, dt=None):
        tendencies = {
            "x_velocity": self.get_validation_field(
                raw_state_np, "x_velocity"
            ),
            "y_velocity": self.get_validation_field(
                raw_state_np, "y_velocity"
            ),
        }
        diagnostics = {}
        return tendencies, diagnostics

    def get_validation_field(self, raw_state_np, name):
        dx = self.ds.grid.dx.to_units("m").values.item()
        dy = self.ds.grid.dy.to_units("m").values.item()
        phi = raw_state_np[name]
        if self.ds.grid.nx < 3:
            return self.smooth_coeff * second_order_diffusion_yz(dy, phi)
        elif self.ds.grid.ny < 3:
            return self.smooth_coeff * second_order_diffusion_xz(dx, phi)
        else:
            return self.smooth_coeff * second_order_diffusion_xyz(dx, dy, phi)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_second_order(data, backend, dtype):
    ds = DomainSuite(
        data,
        backend,
        dtype,
        zaxis_length=(1, 1),
        nb_min=1,
        check_rebuild=False,
    )
    assume(ds.grid.nx > 2 or ds.grid.ny > 2)
    ts = SecondOrder(ds)
    ts.run()


class FourthOrder(SecondOrder):
    @cached_property
    def component(self):
        order = "fourth_order"
        if self.ds.grid.nx < 5:
            order += "_1dy"
        elif self.ds.grid.ny < 5:
            order += "_1dx"

        return BurgersHorizontalDiffusion(
            self.ds.domain,
            self.ds.grid_type,
            order,
            DataArray(self.smooth_coeff, attrs={"units": "m^2 s^-1"}),
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_options=self.ds.so,
        )

    def assert_allclose(self, name, field_a, field_b):
        if self.ds.grid.nx < 5:
            assert_yz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)
        elif self.ds.grid.ny < 5:
            assert_xz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)
        else:
            assert_xyz(field_a, field_b, self.ds.domain.horizontal_boundary.nb)

    def get_validation_field(self, raw_state_np, name):
        dx = self.ds.grid.dx.to_units("m").values.item()
        dy = self.ds.grid.dy.to_units("m").values.item()
        phi = raw_state_np[name]
        if self.ds.grid.nx < 5:
            return self.smooth_coeff * fourth_order_diffusion_yz(dy, phi)
        elif self.ds.grid.ny < 5:
            return self.smooth_coeff * fourth_order_diffusion_xz(dx, phi)
        else:
            return self.smooth_coeff * fourth_order_diffusion_xyz(dx, dy, phi)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_fourth_order(data, backend, dtype):
    ds = DomainSuite(
        data,
        backend,
        dtype,
        zaxis_length=(1, 1),
        nb_min=2,
        check_rebuild=False,
    )
    assume(ds.grid.nx > 4 or ds.grid.ny > 4)
    ts = FourthOrder(ds)
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
