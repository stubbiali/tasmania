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
from typing import Callable, Optional, Sequence, Type, Union
from tasmania.python.framework import protocol as prt
from tasmania.python.framework.options import BackendOptions
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol_utils import (
    Registry,
    multiregister,
    set_runtime_attribute,
)


class StencilSubroutine:
    registry = Registry()

    def __new__(
        cls: Type["StencilSubroutine"], backend: str, stencil: str
    ) -> Callable:
        key = ("stencil_subroutine", backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                "stencil_subroutine",
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
        cls: Type["StencilSubroutine"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        return multiregister(
            handle,
            cls.registry,
            (
                "function",
                "stencil_subroutine",
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


class StencilCompiler:
    registry = Registry()

    def __new__(
        cls: Type["StencilCompiler"],
        backend: str,
        stencil: str,
        *,
        backend_options: Optional[BackendOptions] = None,
    ) -> Callable:
        definition = StencilDefinition(backend, stencil)
        key = ("stencil_compiler", backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                "stencil_compiler",
                "backend",
                backend,
                "stencil",
                stencil,
            )
            return obj(definition, backend_options=backend_options)
        except KeyError:
            raise FactoryRegistryError(
                f"No stencil compiler registered for the backend '{backend}'."
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
                "stencil_compiler",
                "backend",
                backend,
                "stencil",
                stencil,
            ),
        )

    @staticmethod
    def template(definition: Callable, *, backend_options: BackendOptions):
        pass


# lower-case aliases
stencil_subroutine = StencilSubroutine
stencil_definition = StencilDefinition
stencil_compiler = StencilCompiler
