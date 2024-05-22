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
from sympl._core.static_operators import StaticComponentOperator


class StaticOperator:
    @classmethod
    def get_tendency_properties(cls, promoter):
        input_properties = StaticComponentOperator.factory(
            "input_properties"
        ).get_properties_with_dims(promoter)
        out = {}
        for name in input_properties:
            tendency_name = input_properties[name].pop(
                "tendency_name", name.replace("tendency_of_", "")
            )
            out[tendency_name] = input_properties[name]
        return out

    @classmethod
    def get_diagnostic_properties(cls, promoter):
        tendency_properties = StaticComponentOperator.factory(
            "input_tendency_properties"
        ).get_properties_with_dims(promoter)
        out = {}
        for name in tendency_properties:
            diagnostic_name = tendency_properties[name].pop(
                "diagnostic_name", "tendency_of_" + name
            )
            out[diagnostic_name] = tendency_properties[name]
        return out
