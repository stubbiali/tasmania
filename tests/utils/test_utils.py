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
import numpy
import pytest

from tasmania.python.utils.utils import feed_module


def test_feed_module():
    from tests.utils import namelist
    from tests.utils import namelist_baseline as baseline

    feed_module(target=namelist, source=baseline)

    assert hasattr(namelist, "bar")
    assert namelist.bar == 1.0
    assert hasattr(namelist, "foo")
    assert namelist.foo is True
    assert hasattr(namelist, "pippo")
    assert namelist.pippo == "Hello, world!"
    assert hasattr(namelist, "franco")
    assert namelist.franco == "Hello, world!"
    assert hasattr(namelist, "ciccio")
    assert namelist.ciccio == numpy.float64


if __name__ == "__main__":
    pytest.main([__file__])
