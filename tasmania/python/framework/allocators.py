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
import itertools
import numpy as np
from nptyping import NDArray
from typing import Any, Callable, Optional, Sequence, Type, Union

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol_utils import (
    multiregister,
    Registry,
    set_runtime_attribute,
)

from tasmania.python.utils import taz_types


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
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[taz_types.dtype_t] = None,
        **kwargs: Any
    ) -> NDArray:
        """Dispatch the call to the proper registered object."""
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
            return obj(shape=shape, dtype=dtype, **kwargs)
        except KeyError:
            raise FactoryRegistryError(
                f"No allocator registered for the backend '{backend}'."
            )

    @classmethod
    def register(
        cls: Type["Allocator"],
        backend: Union[str, Sequence[str]],
        stencil: Union[str, Sequence[str]] = prt.wildcard,
        handle: Optional[Callable] = None,
    ) -> Callable:
        """Decorator to register an object."""
        return multiregister(
            cls.registry,
            handle,
            "function",
            cls.function,
            "backend",
            backend,
            "stencil",
            stencil,
        )

    @staticmethod
    def template(shape: Sequence[int], dtype: taz_types.dtype_t, **kwargs):
        """Signature template for any registered object."""
        pass


class Empty(Allocator):
    """A class to centrally manage objects allocating empty storages."""

    function = "empty"


class Ones(Allocator):
    """A class to centrally manage objects allocating storages filled with ones."""

    function = "ones"


class Zeros(Allocator):
    """A class to centrally manage objects allocating storages filled with zeros."""

    function = "zeros"


# numpy-compliant aliases
empty = Empty
ones = Ones
zeros = Zeros
