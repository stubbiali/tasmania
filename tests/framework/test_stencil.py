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
import numpy as np
import pytest

from tasmania.third_party import cupy, gt4py, numba

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.options import BackendOptions
from tasmania.python.framework.stencil import (
    StencilCompiler,
    StencilDefinition,
    SubroutineDefinition,
    StencilFactory,
)
from tasmania.python.framework.subclasses.stencil_compilers import (
    compiler_numpy,
)

if gt4py:
    from tasmania.python.framework.subclasses.stencil_compilers import (
        compiler_gt4py,
    )

from tests.framework.test_allocators import TestEmpty, TestOnes, TestZeros


class TestStencilSubroutine:
    @staticmethod
    def check_registry_keys(r):
        f = "stencil_subroutine"
        backends = (
            "numpy",
            "cupy" if cupy else "numpy",
            "gt4py*" if gt4py else "numpy",
        )

        assert f in r
        assert all(backend in r[f] for backend in backends)
        assert all("absolute" in r[f][backend] for backend in backends)
        assert all("laplacian" in r[f][backend] for backend in backends)
        assert all("negative" in r[f][backend] for backend in backends)
        assert all("positive" in r[f][backend] for backend in backends)

    def test_registry_keys(self):
        self.check_registry_keys(SubroutineDefinition.registry)

    @staticmethod
    def check_registry_values(r):
        f = "stencil_subroutine"

        from tasmania.python.framework.subclasses.subroutine_definitions import (
            laplacian,
            math,
        )

        # absolute
        assert r[f]["numpy"]["absolute"] == math.absolute_numpy
        if cupy:
            assert r[f]["cupy"]["absolute"] == math.absolute_cupy
        if gt4py:
            assert r[f]["gt4py*"]["absolute"] == math.absolute_gt4py

        # laplacian
        assert r[f]["numpy"]["laplacian"] == laplacian.laplacian_numpy
        if cupy:
            assert r[f]["cupy"]["laplacian"] == laplacian.laplacian_numpy
        if gt4py:
            assert r[f]["gt4py*"]["laplacian"] == laplacian.laplacian_gt4py

        # negative
        assert r[f]["numpy"]["negative"] == math.negative_numpy
        if cupy:
            assert r[f]["cupy"]["negative"] == math.negative_cupy
        if gt4py:
            assert r[f]["gt4py*"]["negative"] == math.negative_gt4py

        # positive
        assert r[f]["numpy"]["positive"] == math.positive_numpy
        if cupy:
            assert r[f]["cupy"]["positive"] == math.positive_cupy
        if gt4py:
            assert r[f]["gt4py*"]["positive"] == math.positive_gt4py

    def test_registry_values(self):
        self.check_registry_values(SubroutineDefinition.registry)

    @staticmethod
    def check_factory(s):
        from tasmania.python.framework.subclasses.subroutine_definitions import (
            laplacian,
            math,
        )

        # absolute
        assert s("numpy", "absolute") == math.absolute_numpy
        if cupy:
            assert s("cupy", "absolute") == math.absolute_cupy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "absolute") == math.absolute_gt4py
                )

        # laplacian
        assert s("numpy", "laplacian") == laplacian.laplacian_numpy
        if cupy:
            assert s("cupy", "laplacian") == laplacian.laplacian_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "laplacian")
                    == laplacian.laplacian_gt4py
                )

        # negative
        assert s("numpy", "negative") == math.negative_numpy
        if cupy:
            assert s("cupy", "negative") == math.negative_cupy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "negative") == math.negative_gt4py
                )

        # positive
        assert s("numpy", "positive") == math.positive_numpy
        if cupy:
            assert s("cupy", "positive") == math.positive_cupy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "positive") == math.positive_gt4py
                )

    def test_factory(self):
        self.check_factory(SubroutineDefinition)


class TestStencilDefinition:
    @staticmethod
    def check_registry_keys(r):
        f = "stencil_definition"
        backends = (
            "numpy",
            "cupy" if cupy else "numpy",
            "gt4py*" if gt4py else "numpy",
        )

        assert f in r
        assert all(backend in r[f] for backend in backends)

        # algorithm.py
        assert all("irelax" in r[f][backend] for backend in backends)
        assert all("relax" in r[f][backend] for backend in backends)
        assert all("sts_rk2_0" in r[f][backend] for backend in backends)
        assert all("sts_rk3ws_0" in r[f][backend] for backend in backends)

        # cla.py
        assert all("thomas" in r[f][backend] for backend in backends)

        # copy.py
        assert all("copy" in r[f][backend] for backend in backends)
        assert all("copychange" in r[f][backend] for backend in backends)

        # diffusion.py
        assert all("diffusion" in r[f][backend] for backend in backends)

        # math.py
        assert all("abs" in r[f][backend] for backend in backends)
        assert all("add" in r[f][backend] for backend in backends)
        assert all("addsub" in r[f][backend] for backend in backends)
        assert all("clip" in r[f][backend] for backend in backends)
        assert all("fma" in r[f][backend] for backend in backends)
        assert all("iabs" in r[f][backend] for backend in backends)
        assert all("iadd" in r[f][backend] for backend in backends)
        assert all("iaddsub" in r[f][backend] for backend in backends)
        assert all("iclip" in r[f][backend] for backend in backends)
        assert all("imul" in r[f][backend] for backend in backends)
        assert all("iscale" in r[f][backend] for backend in backends)
        assert all("isub" in r[f][backend] for backend in backends)
        assert all("mul" in r[f][backend] for backend in backends)
        assert all("scale" in r[f][backend] for backend in backends)
        assert all("sub" in r[f][backend] for backend in backends)

    def test_registry_keys(self):
        self.check_registry_keys(StencilDefinition.registry)

    @staticmethod
    def check_registry_values(r):
        r = StencilDefinition.registry
        f = "stencil_definition"

        from tasmania.python.framework.subclasses.stencil_definitions import (
            algorithms,
            cla,
            copy,
            diffusion,
            math,
        )

        # algorithms::irelax
        assert r[f]["numpy"]["irelax"] == algorithms.irelax_numpy
        if cupy:
            assert r[f]["cupy"]["irelax"] == algorithms.irelax_numpy
        if gt4py:
            assert r[f]["gt4py*"]["irelax"] == algorithms.irelax_gt4py

        # algorithms::relax
        assert r[f]["numpy"]["relax"] == algorithms.relax_numpy
        if cupy:
            assert r[f]["cupy"]["relax"] == algorithms.relax_numpy
        if gt4py:
            assert r[f]["gt4py*"]["relax"] == algorithms.relax_gt4py

        # algorithms::sts_rk2_0
        assert r[f]["numpy"]["sts_rk2_0"] == algorithms.sts_rk2_0_numpy
        if cupy:
            assert r[f]["cupy"]["sts_rk2_0"] == algorithms.sts_rk2_0_numpy
        if gt4py:
            assert r[f]["gt4py*"]["sts_rk2_0"] == algorithms.sts_rk2_0_gt4py

        # algorithms::sts_rk3ws_0
        assert r[f]["numpy"]["sts_rk3ws_0"] == algorithms.sts_rk3ws_0_numpy
        if cupy:
            assert r[f]["cupy"]["sts_rk3ws_0"] == algorithms.sts_rk3ws_0_numpy
        if gt4py:
            assert (
                r[f]["gt4py*"]["sts_rk3ws_0"] == algorithms.sts_rk3ws_0_gt4py
            )

        # cla::thomas
        assert r[f]["numpy"]["thomas"] == cla.thomas_numpy
        if cupy:
            assert r[f]["cupy"]["thomas"] == cla.thomas_numpy
        if gt4py:
            assert r[f]["gt4py*"]["thomas"] == cla.thomas_gt4py

        # copy::copy
        assert r[f]["numpy"]["copy"] == copy.copy_numpy
        if cupy:
            assert r[f]["cupy"]["copy"] == copy.copy_numpy
        if gt4py:
            assert r[f]["gt4py*"]["copy"] == copy.copy_gt4py

        # copy::copychange
        assert r[f]["numpy"]["copychange"] == copy.copychange_numpy
        if cupy:
            assert r[f]["cupy"]["copychange"] == copy.copychange_numpy
        if gt4py:
            assert r[f]["gt4py*"]["copychange"] == copy.copychange_gt4py

        # diffusion::diffusion
        assert r[f]["numpy"]["diffusion"] == diffusion.diffusion_numpy
        if cupy:
            assert r[f]["cupy"]["diffusion"] == diffusion.diffusion_numpy
        if gt4py:
            assert r[f]["gt4py*"]["diffusion"] == diffusion.diffusion_gt4py

        # math::abs
        assert r[f]["numpy"]["abs"] == math.abs_numpy
        if cupy:
            assert r[f]["cupy"]["abs"] == math.abs_numpy
        if gt4py:
            assert r[f]["gt4py*"]["abs"] == math.abs_gt4py

        # math::add
        assert r[f]["numpy"]["add"] == math.add_numpy
        if cupy:
            assert r[f]["cupy"]["add"] == math.add_numpy
        if gt4py:
            assert r[f]["gt4py*"]["add"] == math.add_gt4py

        # math::addsub
        assert r[f]["numpy"]["addsub"] == math.addsub_numpy
        if cupy:
            assert r[f]["cupy"]["addsub"] == math.addsub_numpy
        if gt4py:
            assert r[f]["gt4py*"]["addsub"] == math.addsub_gt4py

        # math::clip
        assert r[f]["numpy"]["clip"] == math.clip_numpy
        if cupy:
            assert r[f]["cupy"]["clip"] == math.clip_cupy
        if gt4py:
            assert r[f]["gt4py*"]["clip"] == math.clip_gt4py

        # math::fma
        assert r[f]["numpy"]["fma"] == math.fma_numpy
        if cupy:
            assert r[f]["cupy"]["fma"] == math.fma_numpy
        if gt4py:
            assert r[f]["gt4py*"]["fma"] == math.fma_gt4py

        # math::iabs
        assert r[f]["numpy"]["iabs"] == math.iabs_numpy
        if cupy:
            assert r[f]["cupy"]["iabs"] == math.iabs_numpy
        if gt4py:
            assert r[f]["gt4py*"]["iabs"] == math.iabs_gt4py

        # math::iadd
        assert r[f]["numpy"]["iadd"] == math.iadd_numpy
        if cupy:
            assert r[f]["cupy"]["iadd"] == math.iadd_numpy
        if gt4py:
            assert r[f]["gt4py*"]["iadd"] == math.iadd_gt4py

        # math::iaddsub
        assert r[f]["numpy"]["iaddsub"] == math.iaddsub_numpy
        if cupy:
            assert r[f]["cupy"]["iaddsub"] == math.iaddsub_numpy
        if gt4py:
            assert r[f]["gt4py*"]["iaddsub"] == math.iaddsub_gt4py

        # math::iclip
        assert r[f]["numpy"]["iclip"] == math.iclip_numpy
        if cupy:
            assert r[f]["cupy"]["iclip"] == math.iclip_cupy
        if gt4py:
            assert r[f]["gt4py*"]["iclip"] == math.iclip_gt4py

        # math::imul
        assert r[f]["numpy"]["imul"] == math.imul_numpy
        if cupy:
            assert r[f]["cupy"]["imul"] == math.imul_numpy
        if gt4py:
            assert r[f]["gt4py*"]["imul"] == math.imul_gt4py

        # math::iscale
        assert r[f]["numpy"]["iscale"] == math.iscale_numpy
        if cupy:
            assert r[f]["cupy"]["iscale"] == math.iscale_numpy
        if gt4py:
            assert r[f]["gt4py*"]["iscale"] == math.iscale_gt4py

        # math::isub
        assert r[f]["numpy"]["isub"] == math.isub_numpy
        if cupy:
            assert r[f]["cupy"]["isub"] == math.isub_numpy
        if gt4py:
            assert r[f]["gt4py*"]["isub"] == math.isub_gt4py

        # math::mul
        assert r[f]["numpy"]["mul"] == math.mul_numpy
        if cupy:
            assert r[f]["cupy"]["mul"] == math.mul_numpy
        if gt4py:
            assert r[f]["gt4py*"]["mul"] == math.mul_gt4py

        # math::scale
        assert r[f]["numpy"]["scale"] == math.scale_numpy
        if cupy:
            assert r[f]["cupy"]["scale"] == math.scale_numpy
        if gt4py:
            assert r[f]["gt4py*"]["scale"] == math.scale_gt4py

        # math::sub
        assert r[f]["numpy"]["sub"] == math.sub_numpy
        if cupy:
            assert r[f]["cupy"]["sub"] == math.sub_numpy
        if gt4py:
            assert r[f]["gt4py*"]["sub"] == math.sub_gt4py

    def test_registry_values(self):
        self.check_registry_values(StencilDefinition.registry)

    @staticmethod
    def check_factory(s):
        from tasmania.python.framework.subclasses.stencil_definitions import (
            algorithms,
            cla,
            copy,
            diffusion,
            math,
        )

        # algorithms::irelax
        assert s("numpy", "irelax") == algorithms.irelax_numpy
        if cupy:
            assert s("cupy", "irelax") == algorithms.irelax_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "irelax")
                    == algorithms.irelax_gt4py
                )

        # algorithms::relax
        assert s("numpy", "relax") == algorithms.relax_numpy
        if cupy:
            assert s("cupy", "relax") == algorithms.relax_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "relax") == algorithms.relax_gt4py
                )

        # algorithms::sts_rk2_0
        assert s("numpy", "sts_rk2_0") == algorithms.sts_rk2_0_numpy
        if cupy:
            assert s("cupy", "sts_rk2_0") == algorithms.sts_rk2_0_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "sts_rk2_0")
                    == algorithms.sts_rk2_0_gt4py
                )

        # algorithms::sts_rk3ws_0
        assert s("numpy", "sts_rk3ws_0") == algorithms.sts_rk3ws_0_numpy
        if cupy:
            assert s("cupy", "sts_rk3ws_0") == algorithms.sts_rk3ws_0_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "sts_rk3ws_0")
                    == algorithms.sts_rk3ws_0_gt4py
                )

        # cla::thomas
        assert s("numpy", "thomas") == cla.thomas_numpy
        if cupy:
            assert s("cupy", "thomas") == cla.thomas_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "thomas") == cla.thomas_gt4py

        # copy::copy
        assert s("numpy", "copy") == copy.copy_numpy
        if cupy:
            assert s("cupy", "copy") == copy.copy_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "copy") == copy.copy_gt4py

        # copy::copychange
        assert s("numpy", "copychange") == copy.copychange_numpy
        if cupy:
            assert s("cupy", "copychange") == copy.copychange_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "copychange")
                    == copy.copychange_gt4py
                )

        # diffusion::diffusion
        assert s("numpy", "diffusion") == diffusion.diffusion_numpy
        if cupy:
            assert s("cupy", "diffusion") == diffusion.diffusion_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "diffusion")
                    == diffusion.diffusion_gt4py
                )

        # math::abs
        assert s("numpy", "abs") == math.abs_numpy
        if cupy:
            assert s("cupy", "abs") == math.abs_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "abs") == math.abs_gt4py

        # math::add
        assert s("numpy", "add") == math.add_numpy
        if cupy:
            assert s("cupy", "add") == math.add_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "add") == math.add_gt4py

        # math::addsub
        assert s("numpy", "addsub") == math.addsub_numpy
        if cupy:
            assert s("cupy", "addsub") == math.addsub_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "addsub") == math.addsub_gt4py

        # math::clip
        assert s("numpy", "clip") == math.clip_numpy
        if cupy:
            assert s("cupy", "clip") == math.clip_cupy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "clip") == math.clip_gt4py

        # math::fma
        assert s("numpy", "fma") == math.fma_numpy
        if cupy:
            assert s("cupy", "fma") == math.fma_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "fma") == math.fma_gt4py

        # math::iabs
        assert s("numpy", "iabs") == math.iabs_numpy
        if cupy:
            assert s("cupy", "iabs") == math.iabs_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "iabs") == math.iabs_gt4py

        # math::iadd
        assert s("numpy", "iadd") == math.iadd_numpy
        if cupy:
            assert s("cupy", "iadd") == math.iadd_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "iadd") == math.iadd_gt4py

        # math::iaddsub
        assert s("numpy", "iaddsub") == math.iaddsub_numpy
        if cupy:
            assert s("cupy", "iaddsub") == math.iaddsub_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert (
                    s(f"gt4py:{gt_backend}", "iaddsub") == math.iaddsub_gt4py
                )

        # math::iclip
        assert s("numpy", "iclip") == math.iclip_numpy
        if cupy:
            assert s("cupy", "iclip") == math.iclip_cupy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "iclip") == math.iclip_gt4py

        # math::imul
        assert s("numpy", "imul") == math.imul_numpy
        if cupy:
            assert s("cupy", "imul") == math.imul_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "imul") == math.imul_gt4py

        # math::iscale
        assert s("numpy", "iscale") == math.iscale_numpy
        if cupy:
            assert s("cupy", "iscale") == math.iscale_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "iscale") == math.iscale_gt4py

        # math::isub
        assert s("numpy", "isub") == math.isub_numpy
        if cupy:
            assert s("cupy", "isub") == math.isub_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "isub") == math.isub_gt4py

        # math::mul
        assert s("numpy", "mul") == math.mul_numpy
        if cupy:
            assert s("cupy", "mul") == math.mul_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "mul") == math.mul_gt4py

        # math::scale
        assert s("numpy", "scale") == math.scale_numpy
        if cupy:
            assert s("cupy", "scale") == math.scale_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "scale") == math.scale_gt4py

        # math::sub
        assert s("numpy", "sub") == math.sub_numpy
        if cupy:
            assert s("cupy", "sub") == math.sub_numpy
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert s(f"gt4py:{gt_backend}", "sub") == math.sub_gt4py

    def test_factory(self):
        self.check_factory(StencilDefinition)


class TestStencilCompiler:
    @staticmethod
    def check_registry_keys(r):
        f = "stencil_compiler"
        backends = (
            "numpy",
            "cupy" if cupy else "numpy",
            "gt4py*" if gt4py else "numpy",
        )

        assert f in r
        assert all(backend in r[f] for backend in backends)
        assert all(prt.wildcard in r[f][backend] for backend in backends)

    def test_registry_keys(self):
        self.check_registry_keys(StencilCompiler.registry)

    @staticmethod
    def check_registry_values(r):
        f = "stencil_compiler"

        assert r[f]["numpy"]["ABCDE"] == compiler_numpy
        if cupy:
            assert r[f]["cupy"][prt.wildcard] == compiler_numpy
        if gt4py:
            assert r[f]["gt4py*"]["abcde"] == compiler_gt4py

    def test_registry_values(self):
        self.check_registry_values(StencilCompiler.registry)

    @staticmethod
    def check_factory(s):
        bo = BackendOptions(dtypes={"dtype": float})

        from tasmania.python.framework.subclasses.stencil_definitions import (
            diffusion,
        )

        # assert (
        #     s("diffusion", "numpy", backend_options=bo).func
        #     == diffusion.diffusion_numpy
        # )
        if cupy:
            assert (
                s("diffusion", "cupy", backend_options=bo)
                == diffusion.diffusion_numpy
            )
        if gt4py:
            for gt_backend in ("debug", "numpy", "gtx86", "gtmc"):
                assert isinstance(
                    s("diffusion", f"gt4py:{gt_backend}", backend_options=bo),
                    gt4py.StencilObject,
                )

    def test_factory(self):
        self.check_factory(StencilCompiler)


class TestStencilFactory:
    def test_default_registry_keys(self):
        sf = StencilFactory()

        TestStencilCompiler.check_registry_keys(
            sf._default_stencil_compiler_registry
        )
        TestStencilDefinition.check_registry_keys(
            sf._default_stencil_definition_registry
        )
        TestEmpty().check_registry_keys(
            sf._default_allocator_registry, "empty"
        )
        TestOnes().check_registry_keys(sf._default_allocator_registry, "ones")
        TestZeros().check_registry_keys(
            sf._default_allocator_registry, "zeros"
        )

    def test_default_registry_values(self):
        sf = StencilFactory()

        TestStencilCompiler.check_registry_values(
            sf._default_stencil_compiler_registry
        )
        TestStencilDefinition.check_registry_values(
            sf._default_stencil_definition_registry
        )
        TestEmpty().check_registry_keys(
            sf._default_allocator_registry, "empty"
        )
        TestOnes().check_registry_keys(sf._default_allocator_registry, "ones")
        TestZeros().check_registry_keys(
            sf._default_allocator_registry, "zeros"
        )

    def test_default_factory(self):
        sf = StencilFactory()

        TestStencilCompiler.check_factory(sf.compile_stencil)
        TestEmpty().check_factory(sf.empty)
        TestOnes().check_factory(sf.ones)
        TestZeros().check_factory(sf.zeros)

    def test_empty_registry(self):
        sf = StencilFactory()
        assert len(sf._registry) == 0


if __name__ == "__main__":
    pytest.main([__file__])
