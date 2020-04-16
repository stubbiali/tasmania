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
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import numpy as np
import pytest

from tasmania.python.domain.horizontal_boundaries.dirichlet import (
    Dirichlet,
    Dirichlet1DX,
    Dirichlet1DY,
    dispatch as dispatch_dirichlet,
)
from tasmania.python.domain.horizontal_boundaries.identity import (
    Identity,
    Identity1DX,
    Identity1DY,
    dispatch as dispatch_identity,
)
from tasmania.python.domain.horizontal_boundaries.periodic import (
    Periodic,
    Periodic1DX,
    Periodic1DY,
    dispatch as dispatch_periodic,
)
from tasmania.python.domain.horizontal_boundaries.relaxed import (
    Relaxed,
    Relaxed1DX,
    Relaxed1DY,
    dispatch as dispatch_relaxed,
)
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary


def test_register():
    # dirichlet
    assert "dirichlet" in HorizontalBoundary.register
    assert HorizontalBoundary.register["dirichlet"] == dispatch_dirichlet

    # identity
    assert "identity" in HorizontalBoundary.register
    assert HorizontalBoundary.register["identity"] == dispatch_identity

    # periodic
    assert "periodic" in HorizontalBoundary.register
    assert HorizontalBoundary.register["periodic"] == dispatch_periodic

    # relaxed
    assert "relaxed" in HorizontalBoundary.register
    assert HorizontalBoundary.register["relaxed"] == dispatch_relaxed


def test_factory():
    # dirichlet
    obj = HorizontalBoundary.factory("dirichlet", 10, 20, 3)
    assert isinstance(obj, Dirichlet)
    obj = HorizontalBoundary.factory("dirichlet", 10, 1, 3)
    assert isinstance(obj, Dirichlet1DX)
    obj = HorizontalBoundary.factory("dirichlet", 1, 20, 3)
    assert isinstance(obj, Dirichlet1DY)

    # identity
    obj = HorizontalBoundary.factory("identity", 10, 20, 3)
    assert isinstance(obj, Identity)
    obj = HorizontalBoundary.factory("identity", 10, 1, 3)
    assert isinstance(obj, Identity1DX)
    obj = HorizontalBoundary.factory("identity", 1, 20, 3)
    assert isinstance(obj, Identity1DY)

    # periodic
    obj = HorizontalBoundary.factory("periodic", 10, 20, 3)
    assert isinstance(obj, Periodic)
    obj = HorizontalBoundary.factory("periodic", 10, 1, 3)
    assert isinstance(obj, Periodic1DX)
    obj = HorizontalBoundary.factory("periodic", 1, 20, 3)
    assert isinstance(obj, Periodic1DY)

    # relaxed
    obj = HorizontalBoundary.factory("relaxed", 10, 20, 1, nr=2)
    assert isinstance(obj, Relaxed)
    obj = HorizontalBoundary.factory("relaxed", 10, 1, 1, nr=2)
    assert isinstance(obj, Relaxed1DX)
    obj = HorizontalBoundary.factory("relaxed", 1, 20, 1, nr=2)
    assert isinstance(obj, Relaxed1DY)


if __name__ == "__main__":
    pytest.main([__file__])
