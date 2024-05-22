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
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Type,
    Union,
)

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol import (
    Registry,
    multiregister,
    set_runtime_attribute,
)

if TYPE_CHECKING:
    from tasmania.python.framework.options import StorageOptions


class Allocator(abc.ABC):
    """A class to centrally manage objects allocating storages."""

    # the role taken on by the objects in the context of the tasmania protocol
    function: str = None

    # the dictionary of registered objects
    registry = Registry()

    def __new__(
        cls: Type["Allocator"],
        backend: str,
        stencil: str = prt.wildcard,
        *,
        shape: Sequence[int],
        storage_options: Optional["StorageOptions"] = None,
    ) -> Any:
        """Dispatch the call to the proper registered object."""
        key = (cls.function, backend, stencil)
        try:
            obj = cls.registry[key]
        except KeyError:
            raise FactoryRegistryError(f"No allocator registered for the backend '{backend}'.")
        set_runtime_attribute(
            obj,
            "function",
            cls.function,
            "backend",
            backend,
            "stencil",
            stencil,
        )
        return obj(shape, storage_options=storage_options)

    @classmethod
    def register(
        cls: Type["Allocator"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        """Decorator to register an object."""
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
    def template(shape: Sequence[int], *, storage_options: Optional["StorageOptions"] = None):
        """Signature template for any registered object."""


class Empty(Allocator):
    """A class to centrally manage objects allocating empty storages."""

    function = "empty"


class Ones(Allocator):
    """A class to centrally manage objects allocating storages filled with ones."""

    function = "ones"


class Zeros(Allocator):
    """A class to centrally manage objects allocating storages filled with zeros."""

    function = "zeros"


class AsStorage(abc.ABC):
    """A class to centrally manage objects allocating storages."""

    # the dictionary of registered objects
    registry = Registry()

    def __new__(
        cls: Type["AsStorage"],
        backend: str,
        stencil: str = prt.wildcard,
        *,
        data: "Storage",
        storage_options: Optional["StorageOptions"] = None,
    ) -> "Storage":
        """Dispatch the call to the proper registered object."""
        key = ("as_storage", backend, stencil)
        try:
            obj = cls.registry[key]
        except KeyError:
            raise FactoryRegistryError(
                f"No storage converter registered for the backend '{backend}'."
            )
        set_runtime_attribute(
            obj,
            "function",
            "as_storage",
            "backend",
            backend,
            "stencil",
            stencil,
        )
        return obj(data, storage_options=storage_options)

    @classmethod
    def register(
        cls: Type["Allocator"],
        handle: Optional[Callable] = None,
        backend: Union[str, Sequence[str]] = prt.wildcard,
        stencil: Union[str, Sequence[str]] = prt.wildcard,
    ) -> Callable:
        """Decorator to register an object."""
        return multiregister(
            handle,
            cls.registry,
            (
                "function",
                "as_storage",
                "backend",
                backend,
                "stencil",
                stencil,
            ),
        )

    @staticmethod
    def template(shape: Sequence[int], *, storage_options: Optional["StorageOptions"] = None):
        """Signature template for any registered object."""


# numpy-compliant aliases
as_storage = AsStorage
empty = Empty
ones = Ones
zeros = Zeros
