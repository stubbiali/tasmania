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
import collections
import functools
import inspect
import numba
from numba import cuda
import taichi as ti

import gt4py as gt

from tasmania.python.framework.options import BackendOptions
from tasmania.python.framework.stencil import stencil_compiler
from tasmania.python.utils.utils import get_gt_backend


def remove_annotation(parameter: str):
    if ":" not in parameter:
        return parameter

    i = parameter.find(":")
    j = parameter.find("=")
    if j > i:
        return parameter[:i] + parameter[j:]
    else:
        return parameter[:i]


def wrap(func=None):
    """Wrap a callable to accept unused keyword arguments."""

    def core(handle):
        signature = inspect.signature(handle)

        parameters = []
        arguments = []

        for key, value in signature.parameters.items():
            if value.kind == value.VAR_POSITIONAL:
                parameters.append(remove_annotation(str(value)))
                arguments.append(remove_annotation(str(value)))
            elif value.kind == value.VAR_KEYWORD:
                # ignore
                pass
            elif value.kind in (
                value.POSITIONAL_ONLY,
                value.POSITIONAL_OR_KEYWORD,
            ):
                parameters.append(remove_annotation(str(value)))
                arguments.append(key)
            else:  # keyword-only
                if "*" not in parameters:
                    parameters.append("*")
                parameters.append(remove_annotation(str(value)))
                arguments.append(f"{key}={key}")

        wrapped_str = f"""
def wrapped({", ".join(parameters)}, __func__=None, **kwargs):
    return __func__({", ".join(arguments)})
        """
        exec(wrapped_str)

        return functools.partial(locals()["wrapped"], __func__=handle)

    if func is None:
        return core
    else:
        return core(func)


@stencil_compiler.register(backend=("numpy", "cupy"))
def compiler_numpy(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()

    # inject dtype and external symbols into definition scope
    definition.__globals__.update(bo.dtypes)
    definition.__globals__.update(bo.externals)

    # if backend_options is not None:
    #     # inject dtypes and externals
    #     try:
    #         setattr(definition, "__dtypes__", backend_options.dtypes)
    #         setattr(definition, "__externals__", backend_options.externals)
    #     except AttributeError:
    #         # bound attributes do not define the __dict__ attribute
    #         definition.__dict__["__dtypes__"] = backend_options.dtypes
    #         definition.__dict__["__externals__"] = backend_options.externals

    return wrap(definition)


@stencil_compiler.register(backend="gt4py*")
def compiler_gt4py(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()
    backend = compiler_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    backend_opts = bo.backend_opts or {}
    if gt_backend not in ("debug", "numpy"):
        backend_opts.setdefault("verbose", bo.verbose)
    return gt.gtscript.stencil(
        gt_backend,
        definition,
        build_info=bo.build_info,
        dtypes=bo.dtypes,
        externals=bo.externals,
        rebuild=bo.rebuild,
        **backend_opts
    )


@stencil_compiler.register(backend="numba:cpu")
def compiler_numba_cpu(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()

    # gather external symbols
    externals = {}
    for name, value in bo.externals.items():
        if isinstance(value, collections.Callable):
            externals[name] = numba.jit(
                value,
                cache=bo.cache,
                nopython=bo.nopython,
                parallel=bo.parallel,
            )
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

    return wrap(
        numba.jit(
            definition,
            cache=cache,
            fastmath=bo.fastmath,
            inline=bo.inline,
            nopython=bo.nopython,
            parallel=bo.parallel,
        )
    )


@stencil_compiler.register(backend="numba:gpu")
def compiler_numba_cuda(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()

    cache = bo.cache
    if cache and bo.check_rebuild:
        # check if recompilation is needed
        assert hasattr(definition, "__globals__")
        global_symbols = definition.__globals__
        # for key, value in bo.dtypes.items():
        #     if global_symbols.get(key, None) != value:
        #         cache = False
        for key, value in bo.externals.items():
            if global_symbols.get(key, None) != value:
                cache = False

    # inject dtype and external symbols into definition's scope
    definition.__globals__.update(bo.dtypes)
    definition.__globals__.update(bo.externals)

    core = wrap(cuda.jit(definition))

    def wrapper(*args, __bpg__=None, __tpb__=None, **kwargs):
        return core[__bpg__, __tpb__](*args, **kwargs)

    return functools.partial(
        wrapper, __bpg__=bo.blockspergrid, __tpb__=bo.threadsperblock
    )


@stencil_compiler.register(backend="taichi:*")
def compiler_taichi(definition, *, backend_options=None):
    bo = backend_options or BackendOptions()

    # inject dtype and external symbols into definition scope
    definition.__globals__.update(bo.dtypes)
    definition.__globals__.update(bo.externals)

    # return wrap(definition)
    return wrap(ti.kernel(definition))
