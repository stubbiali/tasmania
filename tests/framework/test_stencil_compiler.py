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
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    cp = np

import gt4py as gt

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.options import BackendOptions
from tasmania.python.framework.stencil_compiler import (
    StencilCompiler,
    StencilDefinition,
    StencilSubroutine,
)
from tasmania.python.framework.subclasses.stencil_compilers import (
    compiler_gt4py,
    compiler_numba,
    compiler_numpy,
)
from tasmania.python.framework.subclasses.stencil_definitions.diffusion import (
    diffusion_gt4py,
    diffusion_numpy,
)
from tasmania.python.framework.subclasses.stencil_subroutines.laplacian import (
    laplacian_gt4py,
    laplacian_numpy,
)


class TestStencilSubroutine:
    def test_registry_keys(self):
        r = StencilSubroutine.registry
        f = "stencil_subroutine"
        backends = ("numpy", "cupy", "gt4py*")

        assert f in r
        assert all(backend in r[f] for backend in backends)
        assert all("laplacian" in r[f][backend] for backend in backends)

    def test_registry_values(self):
        r = StencilSubroutine.registry
        f = "stencil_subroutine"

        assert r[f]["numpy"]["laplacian"] == laplacian_numpy
        assert r[f]["cupy"]["laplacian"] == laplacian_numpy
        assert r[f]["gt4py*"]["laplacian"] == laplacian_gt4py

    def test_factory(self):
        s = StencilSubroutine

        assert s("numpy", "laplacian") == laplacian_numpy
        assert s("cupy", "laplacian") == laplacian_numpy
        for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
            assert s(f"gt4py:{gt_backend}", "laplacian") == laplacian_gt4py


class TestStencilDefinition:
    def test_registry_keys(self):
        r = StencilDefinition.registry
        f = "stencil_definition"
        backends = ("numpy", "cupy", "gt4py*")

        assert f in r
        assert all(backend in r[f] for backend in backends)
        assert all("diffusion" in r[f][backend] for backend in backends)

    def test_registry_values(self):
        r = StencilDefinition.registry
        f = "stencil_definition"

        assert r[f]["numpy"]["diffusion"] == diffusion_numpy
        assert r[f]["cupy"]["diffusion"] == diffusion_numpy
        assert r[f]["gt4py*"]["diffusion"] == diffusion_gt4py

    def test_factory(self):
        s = StencilDefinition

        assert s("numpy", "diffusion") == diffusion_numpy
        assert s("cupy", "diffusion") == diffusion_numpy
        for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
            assert s(f"gt4py:{gt_backend}", "diffusion") == diffusion_gt4py


class TestStencilCompiler:
    def test_registry_keys(self):
        r = StencilCompiler.registry
        f = "stencil_compiler"
        backends = ("numpy", "cupy", "numba:cpu", "gt4py*")

        assert f in r
        assert all(backend in r[f] for backend in backends)
        assert all(prt.wildcard in r[f][backend] for backend in backends)

    def test_registry_values(self):
        r = StencilCompiler.registry
        f = "stencil_compiler"

        assert r[f]["numpy"]["ABCDE"] == compiler_numpy
        assert r[f]["cupy"][prt.wildcard] == compiler_numpy
        assert r[f]["numba:cpu"]["01234566789"] == compiler_numba
        assert r[f]["gt4py*"]["abcde"] == compiler_gt4py

    def test_factory(self):
        s = StencilCompiler
        bo = BackendOptions(dtypes={"dtype": float})

        assert s("numpy", "diffusion", backend_options=bo) == diffusion_numpy
        assert s("cupy", "diffusion", backend_options=bo) == diffusion_numpy
        for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
            assert isinstance(
                s(f"gt4py:{gt_backend}", "diffusion", backend_options=bo),
                gt.StencilObject,
            )


if __name__ == "__main__":
    pytest.main([__file__])
