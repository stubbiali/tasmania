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
import inspect
from nptyping import NDArray
from typing import Callable, Optional, Sequence

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.allocators import Allocator
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil_compiler import (
    StencilDefinition,
    StencilCompiler,
)
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol_utils import (
    Registry,
    multiregister,
    set_runtime_attribute,
)


class StencilFactory(abc.ABC):
    default_allocator_registry = Allocator.registry
    default_definition_registry = StencilDefinition.registry
    default_compiler_registry = StencilCompiler.registry

    def __init__(
        self: "StencilFactory",
        *,
        backend_options: Optional[BackendOptions] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        self.backend_options = backend_options or BackendOptions()
        self.storage_options = storage_options or StorageOptions()

        self.registry = Registry()
        self._fill_registry()

    def compile(
        self: "StencilFactory",
        backend: str,
        stencil: str,
        *,
        backend_options: Optional[BackendOptions] = None
    ) -> Callable:
        definition_key = ("stencil_definition", backend, stencil)
        try:
            definition = self.registry[definition_key]
        except KeyError:
            try:
                definition = self.default_definition_registry[definition_key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No definition of the stencil '{stencil}' found for the "
                    f"backend '{backend}'."
                )

        compiler_key = ("stencil_compiler", backend, stencil)
        try:
            compiler = self.registry[compiler_key]
        except KeyError:
            try:
                compiler = self.default_compiler_registry[compiler_key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No compiler found for the backend '{backend}'."
                )

        set_runtime_attribute(
            compiler,
            "function",
            "stencil_compiler",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        bo = backend_options or self.backend_options

        return compiler(definition, backend_options=bo)

    def empty(
        self: "StencilFactory",
        backend: str,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions]
    ) -> NDArray:
        return self._allocate(
            "empty", backend, stencil, shape, storage_options
        )

    def ones(
        self: "StencilFactory",
        backend: str,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions] = None
    ) -> NDArray:
        return self._allocate("ones", backend, stencil, shape, storage_options)

    def zeros(
        self: "StencilFactory",
        backend: str,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions] = None
    ) -> NDArray:
        return self._allocate(
            "zeros", backend, stencil, shape, storage_options
        )

    def _allocate(
        self: "StencilFactory",
        function: str,
        backend: str,
        stencil: str,
        shape: Sequence[int],
        storage_options: StorageOptions,
    ) -> NDArray:
        key = (function, backend, stencil)

        try:
            allocator = self.registry[key]
        except KeyError:
            try:
                allocator = self.default_allocator_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No allocator registered for the backend '{backend}'."
                )

        so = storage_options or self.storage_options

        return allocator(shape, storage_options=so)

    def _fill_registry(self: "StencilFactory") -> None:
        methods = inspect.getmembers(
            self,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        )
        for (name, handle) in methods:
            if getattr(handle, prt.attribute, None) is not None:
                prt_dict = getattr(handle, prt.attribute)
                args = tuple(
                    item for pair in prt_dict.items() for item in pair
                )
                multiregister(handle, self.registry, args)
