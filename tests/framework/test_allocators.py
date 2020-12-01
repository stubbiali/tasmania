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
import abc
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    cp = np

import gt4py as gt

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.allocators import empty, ones, zeros
from tasmania.python.framework.options import StorageOptions
from tasmania.python.framework.subclasses.allocators.empty import (
    empty_cupy,
    empty_gt4py,
    empty_numpy,
)
from tasmania.python.framework.subclasses.allocators.ones import (
    ones_cupy,
    ones_gt4py,
    ones_numpy,
)
from tasmania.python.framework.subclasses.allocators.zeros import (
    zeros_cupy,
    zeros_gt4py,
    zeros_numpy,
)


class _TestAllocator(abc.ABC):
    subclass = None
    function = None
    backends = ("numpy", "numba:cpu", "cupy", "numba:gpu", "gt4py*")
    values = {}

    def test_registry_keys(self):
        r = self.subclass.registry
        f = self.function
        assert f in r
        assert all(backend in r[f] for backend in self.backends)
        assert all(prt.wildcard in r[f][backend] for backend in self.backends)

    def test_registry_values(self):
        r = self.subclass.registry
        f = self.function
        assert all(
            r[f][backend][prt.wildcard] == self.values[backend]
            for backend in self.backends
        )

    def test_factory(self):
        s = self.subclass
        shape = (3, 4, 5)
        so = StorageOptions()

        for backend in ("numpy", "numba:cpu"):
            obj = s(backend=backend, shape=shape, storage_options=so)
            assert isinstance(obj, np.ndarray)
            assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
            assert obj.dtype == so.dtype

        for backend in ("cupy", "numba:gpu"):
            obj = s(backend=backend, shape=shape, storage_options=so)
            assert isinstance(obj, cp.ndarray)
            assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
            assert obj.dtype == so.dtype

        for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
            obj = s(
                backend=f"gt4py:{gt_backend}", shape=shape, storage_options=so
            )
            assert isinstance(obj, gt.storage.storage.Storage)
            assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
            assert obj.dtype == so.dtype
            assert obj.backend == gt_backend


class TestEmpty(_TestAllocator):
    subclass = empty
    function = "empty"
    values = {
        "numpy": empty_numpy,
        "numba:cpu": empty_numpy,
        "cupy": empty_cupy,
        "numba:gpu": empty_cupy,
        "gt4py*": empty_gt4py,
    }


class TestOnes(_TestAllocator):
    subclass = ones
    function = "ones"
    values = {
        "numpy": ones_numpy,
        "numba:cpu": ones_numpy,
        "cupy": ones_cupy,
        "numba:gpu": ones_cupy,
        "gt4py*": ones_gt4py,
    }


class TestZeros(_TestAllocator):
    subclass = zeros
    function = "zeros"
    values = {
        "numpy": zeros_numpy,
        "numba:cpu": zeros_numpy,
        "cupy": zeros_cupy,
        "numba:gpu": zeros_cupy,
        "gt4py*": zeros_gt4py,
    }


if __name__ == "__main__":
    pytest.main([__file__])
