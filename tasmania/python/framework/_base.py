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
"""Base components placed in a separate file to avoid circular dependencies."""
import abc
from typing import Optional, TYPE_CHECKING

from sympl._core.base_component import BaseComponent

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        PropertyDict,
    )


class BaseConcurrentCoupling(BaseComponent):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BaseFromDiagnosticToTendency(abc.ABC):
    def __init__(self):
        self._initialized = True

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def __call__(
        self,
        diagnostics: "DataArrayDict",
        *,
        out: Optional["DataArrayDict"] = None,
    ) -> "DataArrayDict":
        pass

    @abc.abstractmethod
    def allocate_tendency(self, name: str) -> "NDArrayLike":
        pass

    @abc.abstractmethod
    def array_call(
        self, diagnostics: "NDArrayLikeDict", out: "NDArrayLikeDict"
    ) -> None:
        pass


class BaseDiagnosticComponentComposite(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BaseFromTendencyToDiagnostic(abc.ABC):
    def __init__(self):
        self._initialized = True

    @property
    @abc.abstractmethod
    def input_tendency_properties(self) -> "PropertyDict":
        pass

    @abc.abstractmethod
    def __call__(
        self,
        tendencies: "DataArrayDict",
        *,
        out: Optional["DataArrayDict"] = None
    ) -> "DataArrayDict":
        pass

    @abc.abstractmethod
    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        pass

    @abc.abstractmethod
    def array_call(
        self, tendencies: "NDArrayLikeDict", out: "NDArrayLikeDict"
    ) -> None:
        pass
