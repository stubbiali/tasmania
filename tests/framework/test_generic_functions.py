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
import pytest

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import StorageOptions

from tests.conf import backend as conf_backend, dtype as conf_dtype


@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_to_numpy(backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    shape = (10, 20, 15)
    storage_options = StorageOptions(dtype=dtype)
    data = zeros(backend, shape=shape, storage_options=storage_options)
    data[0, 0, 0] = 1
    data[5, 15, 10] = -2
    data[3, 3, 14] = 3

    # ========================================
    # test bed
    # ========================================
    np_data = to_numpy(data)
    assert all(i == j for i, j in zip(np_data.shape, shape))
    assert np_data.dtype == data.dtype
    assert np_data[0, 0, 0] == 1
    assert np_data[5, 15, 10] == -2
    assert np_data[3, 3, 14] == 3


if __name__ == "__main__":
    pytest.main([__file__])
