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
import abc
import numpy as np
import pytest

from tasmania.third_party import cupy, gt4py, numba

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.allocators import as_storage, empty, ones, zeros
from tasmania.python.framework.options import StorageOptions
from tasmania.python.framework.subclasses.allocators.as_storage_numpy import (
    as_storage_numpy,
)
from tasmania.python.framework.subclasses.allocators.empty import empty_numpy
from tasmania.python.framework.subclasses.allocators.ones import ones_numpy
from tasmania.python.framework.subclasses.allocators.zeros import zeros_numpy

if cupy:
    from tasmania.python.framework.subclasses.allocators.as_storage_cupy import (
        as_storage_cupy,
    )
    from tasmania.python.framework.subclasses.allocators.empty import (
        empty_cupy,
    )
    from tasmania.python.framework.subclasses.allocators.ones import ones_cupy
    from tasmania.python.framework.subclasses.allocators.zeros import (
        zeros_cupy,
    )

if gt4py:
    from tasmania.python.framework.subclasses.allocators.as_storage_gt4py import (
        as_storage_gt4py,
    )
    from tasmania.python.framework.subclasses.allocators.empty import (
        empty_gt4py,
    )
    from tasmania.python.framework.subclasses.allocators.ones import ones_gt4py
    from tasmania.python.framework.subclasses.allocators.zeros import (
        zeros_gt4py,
    )


class _TestAllocator(abc.ABC):
    subclass = None
    function = None
    backends = {
        "numpy",
        "numba:cpu" if numba else "numpy",
        "cupy" if cupy else "numpy",
        "numba:gpu" if cupy and numba else "numpy",
        "gt4py*" if gt4py else "numpy",
    }
    values = {}

    def check_registry_keys(self, r, f):
        assert f in r
        assert all(backend in r[f] for backend in self.backends)
        assert all(prt.wildcard in r[f][backend] for backend in self.backends)

    def test_registry_keys(self):
        self.check_registry_keys(self.subclass.registry, self.function)

    def check_registry_values(self, r, f):
        assert all(
            r[f][backend][prt.wildcard] == self.values[backend]
            for backend in self.backends
        )

    def test_registry_values(self):
        self.check_registry_values(self.subclass.registry, self.function)

    @staticmethod
    def check_factory(s):
        shape = (3, 4, 5)
        so = StorageOptions()

        for backend in ("numpy", "numba:cpu" if numba else "numpy"):
            obj = s(backend=backend, shape=shape, storage_options=so)
            assert isinstance(obj, np.ndarray)
            assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
            assert obj.dtype == so.dtype

        if cupy:
            for backend in ("cupy", "numba:gpu" if numba else "cupy"):
                obj = s(backend=backend, shape=shape, storage_options=so)
                assert isinstance(obj, cupy.ndarray)
                assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
                assert obj.dtype == so.dtype

        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                obj = s(
                    backend=f"gt4py:{gt_backend}",
                    shape=shape,
                    storage_options=so,
                )
                assert isinstance(obj, gt4py.storage.Storage)
                assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
                assert obj.dtype == so.dtype

    def test_factory(self):
        self.check_factory(self.subclass)


class TestEmpty(_TestAllocator):
    subclass = empty
    function = "empty"
    values = {"numpy": empty_numpy}
    if cupy:
        values["cupy"] = empty_cupy
        values["numba:gpu"] = empty_cupy
    if gt4py:
        values["gt4py*"] = empty_gt4py
    if numba:
        values["numba:cpu"] = empty_numpy


class TestOnes(_TestAllocator):
    subclass = ones
    function = "ones"
    values = {"numpy": ones_numpy}
    if cupy:
        values["cupy"] = ones_cupy
        values["numba:gpu"] = ones_cupy
    if gt4py:
        values["gt4py*"] = ones_gt4py
    if numba:
        values["numba:cpu"] = ones_numpy


class TestZeros(_TestAllocator):
    subclass = zeros
    function = "zeros"
    values = {"numpy": zeros_numpy}
    if cupy:
        values["cupy"] = zeros_cupy
        values["numba:gpu"] = zeros_cupy
    if gt4py:
        values["gt4py*"] = zeros_gt4py
    if numba:
        values["numba:cpu"] = zeros_numpy


class TestAsStorage(_TestAllocator):
    subclass = as_storage
    function = "as_storage"
    backends = {
        "numpy",
        "numba:cpu" if numba else "numpy",
        "cupy" if cupy else "numpy",
        "numba:gpu" if cupy and numba else "numpy",
        "gt4py*" if gt4py else "numpy",
    }
    values = {"numpy": as_storage_numpy}
    if numba:
        values["numba:cpu"] = as_storage_numpy
    if cupy:
        values["cupy"] = as_storage_cupy
        if numba:
            values["numba:gpu"] = as_storage_cupy
    if gt4py:
        values["gt4py*"] = as_storage_gt4py

    @staticmethod
    def check_factory():
        backends = {
            "numpy",
            "numba:cpu" if numba else "numpy",
            "cupy" if cupy else "numpy",
            "numba:gpu" if cupy and numba else "numpy",
            "gt4py:debug" if gt4py else "numpy",
            "gt4py:numpy" if gt4py else "numpy",
            "gt4py:gtx86" if gt4py else "numpy",
            "gt4py:gtmc" if gt4py else "numpy",
        }
        shape = (3, 4, 5)
        so = StorageOptions()

        for backend_to in {"numpy", "numba:cpu" if numba else "numpy"}:
            for backend_from in backends:
                data = zeros(backend_from, shape=shape, storage_options=so)
                obj = as_storage(backend_to, data=data)
                assert isinstance(obj, np.ndarray)
                assert all(it1 == it2 for it1, it2 in zip(obj.shape, shape))
                assert obj.dtype == so.dtype

        if cupy:
            for backend_to in {"cupy", "numba:gpu" if numba else "cupy"}:
                for backend_from in backends:
                    data = zeros(backend_from, shape=shape, storage_options=so)
                    obj = as_storage(backend_to, data=data)
                    assert isinstance(obj, cupy.ndarray)
                    assert all(
                        it1 == it2 for it1, it2 in zip(obj.shape, shape)
                    )
                    assert obj.dtype == so.dtype

        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                for backend_from in backends:
                    data = zeros(backend_from, shape=shape, storage_options=so)
                    obj = as_storage(backend=f"gt4py:{gt_backend}", data=data)
                    assert isinstance(obj, gt4py.storage.Storage)
                    assert all(
                        it1 == it2 for it1, it2 in zip(obj.shape, shape)
                    )
                    assert obj.dtype == so.dtype

    def test_factory(self):
        self.check_factory()


if __name__ == "__main__":
    pytest.main([__file__])
