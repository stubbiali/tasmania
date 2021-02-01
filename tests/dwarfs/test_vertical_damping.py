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
    strategies as hyp_st,
)
import pytest

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD
from tasmania.python.dwarfs.subclasses.vertical_dampers import Rayleigh

from tests.strategies import st_domain, st_floats
from tests.utilities import hyp_settings


def test_registry():
    # rayleigh
    assert "rayleigh" in VD.registry
    assert VD.registry["rayleigh"] == Rayleigh


@hyp_settings
@given(hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)
        ),
        label="grid",
    )
    ngrid = domain.numerical_grid
    depth = data.draw(
        hyp_st.integers(min_value=0, max_value=ngrid.nz), label="depth"
    )
    coeff_max = data.draw(
        st_floats(min_value=0, max_value=1e4), label="coeff_max"
    )

    # ========================================
    # test
    # ========================================
    # rayleigh
    obj = VD.factory("rayleigh", ngrid, depth, coeff_max)
    assert isinstance(obj, Rayleigh)


if __name__ == "__main__":
    pytest.main([__file__])
