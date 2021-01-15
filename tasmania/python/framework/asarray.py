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
from typing import Callable, Optional, Sequence, Type, Union

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import FactoryRegistryError
from tasmania.python.utils.protocol import (
    multiregister,
    Registry,
    set_runtime_attribute,
)


class AsArray(abc.ABC):
    """A class to centrally manage objects allocating storages."""

    # the dictionary of registered objects
    registry = Registry()

    def __new__(
        cls: Type["AsArray"], backend: str, stencil: str = prt.wildcard,
    ) -> Callable:
        """Dispatch the call to the proper registered object."""
        key = ("asarray", backend, stencil)
        try:
            obj = cls.registry[key]
            set_runtime_attribute(
                obj,
                "function",
                "asarray",
                "backend",
                backend,
                "stencil",
                stencil,
            )
            return obj()
        except KeyError:
            raise FactoryRegistryError(
                f"No asarray function registered for the backend '{backend}'."
            )

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
            ("function", "asarray", "backend", backend, "stencil", stencil,),
        )

    @staticmethod
    def template():
        """Signature template for any registered object."""
        pass


# lower-case alias
asarray = AsArray
