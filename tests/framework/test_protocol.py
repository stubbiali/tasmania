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
import pytest

from tasmania.python.framework import protocol as prt


def test_attribute_names():
    assert prt.attribute.startswith("__")
    assert prt.attribute.endswith("__")
    assert prt.attribute != prt.runtime_attribute
    assert prt.runtime_attribute.startswith("__")
    assert prt.runtime_attribute.endswith("__")


def test_keys_hierarchy():
    for key in prt.keys_hierarchy:
        assert key in prt.keys
    assert len(prt.keys_hierarchy) == len(prt.keys)


def test_master_key():
    assert prt.master_key == prt.keys_hierarchy[0]


def test_defaults():
    for key in prt.defaults:
        assert key in prt.keys
        assert isinstance(prt.defaults[key], str)


if __name__ == "__main__":
    pytest.main([__file__])
