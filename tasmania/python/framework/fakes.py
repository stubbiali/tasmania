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
from typing import TYPE_CHECKING, Tuple

from tasmania.python.framework.base_components import TendencyComponent
from tasmania.python.utils import taz_types

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


class FakeTendencyComponent(TendencyComponent):
    def __init__(
        self, domain: "Domain", grid_type: str = "numerical", **kwargs
    ) -> None:
        super().__init__(domain, grid_type, **kwargs)

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state
    ) -> Tuple[taz_types.array_dict_t, taz_types.array_dict_t]:
        return {}, {}


class FakeComponent:
    def __init__(
        self, real_component: taz_types.component_t, property_name: str
    ) -> None:
        self.input_properties = getattr(real_component, property_name)
