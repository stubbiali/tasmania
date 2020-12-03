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

from tasmania.python.dwarfs.horizontal_diffusion import (
    HorizontalDiffusion as HD,
)
from tasmania.python.dwarfs.subclasses.horizontal_diffusers import (
    SecondOrder,
    SecondOrder1DX,
    SecondOrder1DY,
    FourthOrder,
    FourthOrder1DX,
    FourthOrder1DY,
)

from tests.strategies import st_floats
from tests.utilities import hyp_settings


def test_registry():
    # second order
    assert "second_order" in HD.registry
    assert HD.registry["second_order"] == SecondOrder
    assert "second_order_1dx" in HD.registry
    assert HD.registry["second_order_1dx"] == SecondOrder1DX
    assert "second_order_1dy" in HD.registry
    assert HD.registry["second_order_1dy"] == SecondOrder1DY

    # fourth order
    assert "fourth_order" in HD.registry
    assert HD.registry["fourth_order"] == FourthOrder
    assert "fourth_order_1dx" in HD.registry
    assert HD.registry["fourth_order_1dx"] == FourthOrder1DX
    assert "fourth_order_1dy" in HD.registry
    assert HD.registry["fourth_order_1dy"] == FourthOrder1DY


@hyp_settings
@given(hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    ni = data.draw(hyp_st.integers(min_value=5, max_value=100), label="ni")
    nj = data.draw(hyp_st.integers(min_value=5, max_value=100), label="nj")
    nk = data.draw(hyp_st.integers(min_value=1, max_value=100), label="nk")
    dx = data.draw(st_floats(min_value=0), label="dx")
    dy = data.draw(st_floats(min_value=0), label="dy")
    diff_coeff = data.draw(st_floats(min_value=0), label="diff_coeff")
    diff_coeff_max = data.draw(
        st_floats(min_value=diff_coeff), label="diff_coeff_max"
    )
    diff_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=nk))

    # ========================================
    # test bed
    # ========================================
    # second_order
    obj = HD.factory(
        "second_order",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, SecondOrder)

    # second_order_1dx
    obj = HD.factory(
        "second_order_1dx",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, SecondOrder1DX)

    # second_order_1dy
    obj = HD.factory(
        "second_order_1dy",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, SecondOrder1DY)

    # fourth_order
    obj = HD.factory(
        "fourth_order",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, FourthOrder)

    # fourth_order_1dx
    obj = HD.factory(
        "fourth_order_1dx",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, FourthOrder1DX)

    # fourth_order_1dy
    obj = HD.factory(
        "fourth_order_1dy",
        (ni, nj, nk),
        dx,
        dy,
        diff_coeff,
        diff_coeff_max,
        diff_damp_depth,
    )
    assert isinstance(obj, FourthOrder1DY)


if __name__ == "__main__":
    pytest.main([__file__])
