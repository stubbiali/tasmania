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
from sympl._core.base_components import TendencyChecker as SymplTendencyChecker

from tasmania.python.utils import typing


class SubsetTendencyChecker(SymplTendencyChecker):
    """
    Ensure that the input dictionary is a *subset* of `tendency_properties`.
    """

    def __init__(self, component: typing.tendency_component_t) -> None:
        super().__init__(component)

    def check_tendencies(
        self, tendency_dict: typing.properties_mapping_t
    ) -> None:
        __tendency_dict = {
            key: value for key, value in tendency_dict.items() if key != "time"
        }
        self._check_extra_tendencies(__tendency_dict)


class SupersetTendencyChecker(SymplTendencyChecker):
    """
    Ensure that the input dictionary is a *superset* of `tendency_properties`.
    """

    def __init__(self, component: typing.tendency_component_t) -> None:
        super().__init__(component)

    def check_tendencies(
        self, tendency_dict: typing.properties_mapping_t
    ) -> None:
        __tendency_dict = {
            key: value for key, value in tendency_dict.items() if key != "time"
        }
        self._check_missing_tendencies(__tendency_dict)
