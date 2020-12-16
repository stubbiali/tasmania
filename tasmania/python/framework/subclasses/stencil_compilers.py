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
# import numba

import gt4py as gt

from tasmania.python.framework.options import BackendOptions
from tasmania.python.framework.stencil import stencil_compiler
from tasmania.python.utils.utils import get_gt_backend


@stencil_compiler.register(backend=("numpy", "cupy"))
def compiler_numpy(definition, *, backend_options=None):
    return definition


@stencil_compiler.register(backend="gt4py*")
def compiler_gt4py(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()
    backend = compiler_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    backend_opts = bo.backend_opts or {}
    return gt.gtscript.stencil(
        gt_backend,
        definition,
        build_info=bo.build_info,
        dtypes=bo.dtypes,
        externals=bo.externals,
        rebuild=bo.rebuild,
        **backend_opts
    )


# @stencil_compiler.register(backend="numba:cpu")
# def compiler_numba(definition, *, backend_options=None):
#     bo = backend_options or BackendOptions()
#     return numba.njit(definition, parallel=bo.parallel)
