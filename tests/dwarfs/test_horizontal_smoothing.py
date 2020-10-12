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

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing as HS
from tasmania.python.dwarfs.subclasses.horizontal_smoothers import (
    FirstOrder,
    FirstOrder1DX,
    FirstOrder1DY,
    SecondOrder,
    SecondOrder1DX,
    SecondOrder1DY,
    ThirdOrder,
    ThirdOrder1DX,
    ThirdOrder1DY,
)

from tests.strategies import st_floats
from tests.utilities import hyp_settings


def test_registry():
    # first order
    assert "first_order" in HS.registry
    assert HS.registry["first_order"] == FirstOrder
    assert "first_order_1dx" in HS.registry
    assert HS.registry["first_order_1dx"] == FirstOrder1DX
    assert "first_order_1dy" in HS.registry
    assert HS.registry["first_order_1dy"] == FirstOrder1DY

    # second order
    assert "second_order" in HS.registry
    assert HS.registry["second_order"] == SecondOrder
    assert "second_order_1dx" in HS.registry
    assert HS.registry["second_order_1dx"] == SecondOrder1DX
    assert "second_order_1dy" in HS.registry
    assert HS.registry["second_order_1dy"] == SecondOrder1DY

    # third order
    assert "third_order" in HS.registry
    assert HS.registry["third_order"] == ThirdOrder
    assert "third_order_1dx" in HS.registry
    assert HS.registry["third_order_1dx"] == ThirdOrder1DX
    assert "third_order_1dy" in HS.registry
    assert HS.registry["third_order_1dy"] == ThirdOrder1DY


@hyp_settings
@given(hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    ni = data.draw(hyp_st.integers(min_value=1, max_value=100), label="ni")
    nj = data.draw(hyp_st.integers(min_value=1, max_value=100), label="nj")
    nk = data.draw(hyp_st.integers(min_value=1, max_value=100), label="nk")
    smooth_coeff = data.draw(st_floats(min_value=0), label="smooth_coeff")
    smooth_coeff_max = data.draw(
        st_floats(min_value=smooth_coeff), label="smooth_coeff_max"
    )
    smooth_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=nk))

    # ========================================
    # test bed
    # ========================================
    # first_order
    obj = HS.factory(
        "first_order", (ni, nj, nk), smooth_coeff, smooth_coeff_max, smooth_damp_depth
    )
    assert isinstance(obj, FirstOrder)

    # first_order_1dx
    obj = HS.factory(
        "first_order_1dx",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, FirstOrder1DX)

    # first_order_1dy
    obj = HS.factory(
        "first_order_1dy",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, FirstOrder1DY)

    # second_order
    obj = HS.factory(
        "second_order", (ni, nj, nk), smooth_coeff, smooth_coeff_max, smooth_damp_depth
    )
    assert isinstance(obj, SecondOrder)

    # second_order_1dx
    obj = HS.factory(
        "second_order_1dx",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, SecondOrder1DX)

    # second_order_1dy
    obj = HS.factory(
        "second_order_1dy",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, SecondOrder1DY)

    # third_order
    obj = HS.factory(
        "third_order", (ni, nj, nk), smooth_coeff, smooth_coeff_max, smooth_damp_depth
    )
    assert isinstance(obj, ThirdOrder)

    # third_order_1dx
    obj = HS.factory(
        "third_order_1dx",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, ThirdOrder1DX)

    # third_order_1dy
    obj = HS.factory(
        "third_order_1dy",
        (ni, nj, nk),
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
    )
    assert isinstance(obj, ThirdOrder1DY)


if __name__ == "__main__":
    pytest.main([__file__])
