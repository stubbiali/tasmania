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
from typing import Callable, Optional, TYPE_CHECKING

from sympl._core.static_operators import StaticComponentOperator

from tasmania.python.framework.static_checkers import (
    check_dims_are_compatible,
    check_units_are_compatible,
)
from tasmania.python.framework.static_operators import merge_dims

if TYPE_CHECKING:
    from sympl._core.typingx import Component, PropertyDict

    from tasmania.python.framework.sequential_update_splitting import (
        SequentialUpdateSplitting,
    )


class StaticOperator:
    input_operator = StaticComponentOperator.factory("input_properties")
    diagnostic_operator = StaticComponentOperator.factory(
        "diagnostic_properties"
    )
    output_operator = StaticComponentOperator.factory("output_properties")

    @classmethod
    def get_input_properties(
        cls, splitter: "SequentialUpdateSplitting"
    ) -> "PropertyDict":
        available = set()
        out = {}

        for component in splitter.components:
            input_properties = cls.input_operator.get_properties_with_dims(
                component
            )
            for name, props in input_properties.items():
                if name in out:
                    check_units_are_compatible(
                        out[name]["units"], props["units"]
                    )
                    check_dims_are_compatible(out[name]["dims"], props["dims"])
                    out[name]["dims"] = merge_dims(
                        out[name]["dims"], props["dims"]
                    )
                elif name not in available:
                    available.add(name)
                    out[name] = props

            diagnostic_properties = cls.diagnostic_operator.get_properties(
                component
            )
            for name in diagnostic_properties:
                available.add(name)

        return out

    @classmethod
    def get_output_properties(
        cls, splitter: "SequentialUpdateSplitting"
    ) -> "PropertyDict":
        out = cls.get_input_properties(splitter)

        for component in splitter.components:
            diagnostic_properties = cls.diagnostic_operator.get_properties(
                component
            )
            out.update(diagnostic_properties)

        return out
