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

import collections

from tasmania.externals import cupy, numba
from tasmania.framework.options import BackendOptions
from tasmania.framework.stencil import subroutine_compiler
from tasmania.framework.subclasses.stencil_compilers import compiler_numpy


subroutine_compiler.register(compiler_numpy, backend="numpy")


if cupy:
    subroutine_compiler.register(compiler_numpy, backend="cupy")


@subroutine_compiler.register(backend="gt4py*")
def compiler_gt4py(definition, *, backend_options=None):
    return definition


if numba:

    @subroutine_compiler.register(backend="numba:cpu*")
    def compiler_numba_cpu(definition, *, backend_options=None):
        bo = backend_options or BackendOptions()

        # gather external symbols
        externals = {}
        for name, value in bo.externals.items():
            if isinstance(value, collections.Callable):
                try:
                    externals[name] = numba.jit(
                        value,
                        cache=bo.cache,
                        nopython=bo.nopython,
                        parallel=bo.parallel,
                    )
                except TypeError:
                    externals[name] = value
            else:
                externals[name] = value

        cache = bo.cache
        if cache and bo.check_rebuild:
            # check if recompilation is needed
            assert hasattr(definition, "__globals__")
            global_symbols = definition.__globals__
            # for key, value in bo.dtypes.items():
            #     if global_symbols.get(key, None) != value:
            #         cache = False
            for key, value in externals.items():
                if global_symbols.get(key, None) != value:
                    cache = False

        # inject dtype and external symbols into definition's scope
        definition.__globals__.update(bo.dtypes)
        definition.__globals__.update(externals)

        return numba.jit(
            definition,
            cache=cache,
            fastmath=bo.fastmath,
            inline=bo.inline,
            nopython=bo.nopython,
            parallel=bo.parallel,
        )
