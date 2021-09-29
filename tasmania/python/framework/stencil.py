# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from typing import Any, Callable, Optional, Sequence, Type, Union

from tasmania.python.framework import protocol as prt
from tasmania.python.framework.allocators import Allocator, AsStorage
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol import (
    Registry,
    multiregister,
    set_runtime_attribute,
)


class SubroutineDefinition:
    registry = Registry()

    def __new__(
        cls: Type["SubroutineDefinition"], backend: str, stencil: str
    ) -> Callable:
        key = ("subroutine_definition", backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                "subroutine_definition",
                "backend",
                backend,
                "stencil",
                stencil,
            )
            return obj
        except KeyError:
            raise FactoryRegistryError(
                f"No subroutine '{stencil}' has been registered "
                f"for the backend '{backend}'."
            )

    @classmethod
    def register(
        cls: Type["SubroutineDefinition"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        return multiregister(
            handle,
            cls.registry,
            (
                "function",
                "subroutine_definition",
                "backend",
                backend,
                "stencil",
                stencil,
            ),
        )

    @staticmethod
    def template(*args, **kwargs):
        pass


class StencilDefinition:
    registry = Registry()

    def __new__(
        cls: Type["StencilDefinition"], backend: str, stencil: str
    ) -> Callable:
        key = ("stencil_definition", backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                "stencil_definition",
                "backend",
                backend,
                "stencil",
                stencil,
            )
            return obj
        except KeyError:
            raise FactoryRegistryError(
                f"No definition of the stencil '{stencil}' has been registered "
                f"for the backend '{backend}'."
            )

    @classmethod
    def register(
        cls: Type["StencilDefinition"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        return multiregister(
            handle,
            cls.registry,
            (
                "function",
                "stencil_definition",
                "backend",
                backend,
                "stencil",
                stencil,
            ),
        )

    @staticmethod
    def template(*args, **kwargs):
        pass


class Compiler:
    definition_factory: Type = None
    function: str = None
    registry: Registry = None

    def __new__(
        cls: Type["StencilCompiler"],
        stencil: str,
        backend: str,
        *,
        backend_options: Optional[BackendOptions] = None,
    ) -> Callable:
        definition = cls.definition_factory(backend, stencil)
        key = (cls.function, backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                cls.function,
                "backend",
                backend,
                "stencil",
                stencil,
            )
            return obj(definition, backend_options=backend_options)
        except KeyError:
            raise FactoryRegistryError(
                f"No compiler registered for the backend '{backend}'."
            )

    @classmethod
    def register(
        cls: Type["StencilCompiler"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        return multiregister(
            handle,
            cls.registry,
            (
                "function",
                cls.function,
                "backend",
                backend,
                "stencil",
                stencil,
            ),
        )

    @staticmethod
    def template(definition: Callable, *, backend_options: BackendOptions):
        pass


class SubroutineCompiler(Compiler):
    definition_factory = SubroutineDefinition
    function = "subroutine_compiler"
    registry = Registry()


class StencilCompiler(Compiler):
    definition_factory = StencilDefinition
    function = "stencil_compiler"
    registry = Registry()


# lower-case aliases
subroutine_definition = SubroutineDefinition
stencil_definition = StencilDefinition
subroutine_compiler = SubroutineCompiler
stencil_compiler = StencilCompiler


class StencilFactory(abc.ABC):
    _default_allocator_registry = Allocator.registry
    _default_as_storage_registry = AsStorage.registry
    _default_stencil_definition_registry = StencilDefinition.registry
    _default_stencil_compiler_registry = StencilCompiler.registry
    _default_subroutine_definition_registry = SubroutineDefinition.registry
    _default_subroutine_compiler_registry = SubroutineCompiler.registry

    def __init__(
        self: "StencilFactory",
        backend: Optional[str] = None,
        backend_options: Optional[BackendOptions] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        self._backend = backend or "numpy"
        self._backend_options = backend_options or BackendOptions()
        self._storage_options = storage_options or StorageOptions()

        self._registry = Registry()
        self._fill_registry()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def backend_options(self) -> BackendOptions:
        return self._backend_options

    @property
    def storage_options(self) -> StorageOptions:
        return self._storage_options

    def as_storage(
        self: "StencilFactory",
        backend: Optional[str] = None,
        stencil: str = prt.wildcard,
        *,
        data: ty.Storage,
        storage_options: Optional[StorageOptions] = None,
    ) -> ty.Storage:
        backend = backend or self.backend
        key = ("as_storage", backend, stencil)
        try:
            as_storage = self._registry[key]
        except KeyError:
            try:
                as_storage = self._default_as_storage_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No storage converter registered for the backend "
                    f"'{backend}'."
                )

        set_runtime_attribute(
            as_storage,
            "function",
            "as_storage",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        so = storage_options or self.storage_options

        return as_storage(data, storage_options=so)

    def compile_stencil(
        self: "StencilFactory",
        stencil: str,
        backend: Optional[str] = None,
        *,
        backend_options: Optional[BackendOptions] = None
    ) -> Callable:
        backend = backend or self.backend
        definition = self.get_stencil_definition(stencil, backend)
        compiler = self.get_stencil_compiler(backend, stencil)
        bo = backend_options or self.backend_options
        return compiler(definition, backend_options=bo)

    def compile_subroutine(
        self: "StencilFactory",
        stencil: str,
        backend: Optional[str] = None,
        *,
        backend_options: Optional[BackendOptions] = None
    ) -> Callable:
        backend = backend or self.backend
        definition = self.get_subroutine_definition(stencil, backend)
        compiler = self.get_subroutine_compiler(backend, stencil)
        bo = backend_options or self.backend_options
        return compiler(definition, backend_options=bo)

    def empty(
        self: "StencilFactory",
        backend: Optional[str] = None,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions] = None
    ) -> ty.Storage:
        return self._allocate(
            "empty", backend or self.backend, stencil, shape, storage_options
        )

    def ones(
        self: "StencilFactory",
        backend: Optional[str] = None,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions] = None
    ) -> ty.Storage:
        return self._allocate(
            "ones", backend or self.backend, stencil, shape, storage_options
        )

    def zeros(
        self: "StencilFactory",
        backend: Optional[str] = None,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional[StorageOptions] = None
    ) -> ty.Storage:
        return self._allocate(
            "zeros", backend or self.backend, stencil, shape, storage_options
        )

    def get_stencil_definition(
        self: "StencilFactory", stencil: str, backend: Optional[str] = None
    ) -> Callable:
        backend = backend or self.backend
        key = ("stencil_definition", backend, stencil)

        try:
            obj = self._registry[key]
        except KeyError:
            try:
                obj = self._default_stencil_definition_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No definition of the stencil '{stencil}' found for the "
                    f"backend '{backend}'."
                )

        set_runtime_attribute(
            obj,
            "function",
            "stencil_definition",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        return obj

    def get_stencil_compiler(
        self: "StencilFactory", backend: str, stencil: Optional[str] = None
    ):
        key = ("stencil_compiler", backend, stencil)

        try:
            obj = self._registry[key]
        except KeyError:
            try:
                obj = self._default_stencil_compiler_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No compiler found for the backend '{backend}'."
                )

        set_runtime_attribute(
            obj,
            "function",
            "stencil_compiler",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        return obj

    def get_subroutine_definition(
        self: "StencilFactory", stencil: str, backend: Optional[str] = None
    ) -> Callable:
        backend = backend or self.backend
        key = ("subroutine_definition", backend, stencil)

        try:
            obj = self._registry[key]
        except KeyError:
            try:
                obj = self._default_subroutine_definition_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No definition of the subroutine '{stencil}' found for "
                    f"the backend '{backend}'."
                )

        set_runtime_attribute(
            obj,
            "function",
            "subroutine_definition",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        return obj

    def get_subroutine_compiler(
        self: "StencilFactory", backend: str, stencil: Optional[str] = None
    ):
        key = ("subroutine_compiler", backend, stencil)

        try:
            obj = self._registry[key]
        except KeyError:
            try:
                obj = self._default_subroutine_compiler_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No compiler found for the backend '{backend}'."
                )

        set_runtime_attribute(
            obj,
            "function",
            "subroutine_compiler",
            "backend",
            backend,
            "stencil",
            stencil,
        )

        return obj

    def _allocate(
        self: "StencilFactory",
        function: str,
        backend: str,
        stencil: str,
        shape: Sequence[int],
        storage_options: StorageOptions,
    ) -> ty.Storage:
        key = (function, backend, stencil)
        try:
            allocator = self._registry[key]
        except KeyError:
            try:
                allocator = self._default_allocator_registry[key]
            except KeyError:
                raise FactoryRegistryError(
                    f"No allocator registered for the backend '{backend}'."
                )

        set_runtime_attribute(
            allocator,
            "function",
            function,
            "backend",
            backend,
            "stencil",
            stencil,
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
                multiregister(handle, self._registry, args)
