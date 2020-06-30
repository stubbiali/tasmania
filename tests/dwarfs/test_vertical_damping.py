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
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
from pandas import Timedelta
import pytest

import gt4py as gt

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD
from tasmania.python.dwarfs.vertical_dampers import Rayleigh
from tasmania.python.utils.storage_utils import zeros

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_domain, st_floats
from tests.utilities import compare_arrays


def test_registry():
    # rayleigh
    assert "rayleigh" in VD.registry
    assert VD.registry["rayleigh"] == Rayleigh


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30)),
        label="grid",
    )
    cgrid = domain.numerical_grid
    depth = data.draw(hyp_st.integers(min_value=0, max_value=cgrid.nz), label="depth")
    coeff_max = data.draw(st_floats(min_value=0, max_value=1e4), label="coeff_max")

    # ========================================
    # test
    # ========================================
    # rayleigh
    obj = VD.factory("rayleigh", cgrid, depth, coeff_max)
    assert isinstance(obj, Rayleigh)


if __name__ == "__main__":
    pytest.main([__file__])
