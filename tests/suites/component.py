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
from datetime import timedelta
from typing import TYPE_CHECKING

from property_cached import cached_property

from tests.strategies import st_timedeltas
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLike

    from tasmania.python.utils.typingx import TimeDelta


class ComponentTestSuite(abc.ABC):
    def __init__(self, domain_suite: DomainSuite) -> None:
        self.ds = domain_suite
        self.hyp_data = self.ds.hyp_data

    @cached_property
    @abc.abstractmethod
    def component(self):
        pass

    @abc.abstractmethod
    def get_state(self) -> "DataArrayDict":
        pass

    def assert_allclose(
        self, name: str, field_a: "NDArrayLike", field_b: "NDArrayLike"
    ) -> None:
        grid_shape = self.component.get_field_grid_shape(name)
        try:
            compare_arrays(
                field_a, field_b, slice=[slice(el) for el in grid_shape]
            )
        except AssertionError:
            raise RuntimeError(f"assert_allclose failed on {name}")

    def get_timestep(self) -> "TimeDelta":
        return self.hyp_data.draw(
            st_timedeltas(
                min_value=timedelta(seconds=0),
                max_value=timedelta(seconds=3600),
            ),
            label="timestep",
        )
