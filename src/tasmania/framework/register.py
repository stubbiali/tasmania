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

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional, TypeVar

    T = TypeVar("T")


def register(
    name: str, registry_class: Optional[type[T]] = None, registry_name: Optional[str] = None
) -> Callable:
    def core(cls):
        rcls = registry_class or cls
        rname = registry_name or "registry"

        if not hasattr(rcls, rname):
            raise RuntimeError(f"Class {rcls.__name__} does not have the attribute '{rname}'.")
        registry = getattr(rcls, rname)

        if name in registry and registry[name] != cls:
            import warnings

            warnings.warn(
                f"Cannot register {cls.__name__} as '{name}' since this name has already been used "
                f"to register {registry[name]}."
            )
        else:
            registry[name] = cls

        return cls

    return core


def factorize(
    name: str, registry_class: type[T], args, kwargs=None, registry_name: Optional[str] = None
) -> T:
    rcls = registry_class
    rname = registry_name or "registry"

    if not hasattr(rcls, rname):
        raise RuntimeError(f"Class {rcls.__name__} does not have the attribute '{rname}'.")
    registry = getattr(rcls, rname)

    if name in registry:
        kwargs = kwargs or {}
        obj = registry[name](*args, **kwargs)
        return obj
    else:
        raise RuntimeError(
            f"No entity has been registered as '{name}'. Available options are: "
            f"{', '.join(key for key in registry.keys())}."
        )
