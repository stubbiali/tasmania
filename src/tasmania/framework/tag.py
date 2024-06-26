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

from tasmania.framework import protocol as prt
from tasmania.utils.protocol import multiregister

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Union


def asarray(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]] = prt.wildcard
) -> Callable:
    return multiregister(args=("function", "asarray", "backend", backend, "stencil", stencil))


def empty(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]] = prt.wildcard
) -> Callable:
    return multiregister(args=("function", "empty", "backend", backend, "stencil", stencil))


def ones(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]] = prt.wildcard
) -> Callable:
    return multiregister(args=("function", "ones", "backend", backend, "stencil", stencil))


def stencil_compiler(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]] = prt.wildcard
) -> Callable:
    return multiregister(
        args=(
            "function",
            "stencil_compiler",
            "backend",
            backend,
            "stencil",
            stencil,
        )
    )


def stencil_definition(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]]
) -> Callable:
    return multiregister(
        args=(
            "function",
            "stencil_definition",
            "backend",
            backend,
            "stencil",
            stencil,
        )
    )


def subroutine_definition(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]]
) -> Callable:
    return multiregister(
        args=(
            "function",
            "subroutine_definition",
            "backend",
            backend,
            "stencil",
            stencil,
        )
    )


def zeros(
    backend: Union[str, Sequence[str]], stencil: Union[str, Sequence[str]] = prt.wildcard
) -> Callable:
    return multiregister(args=("function", "zeros", "backend", backend, "stencil", stencil))
