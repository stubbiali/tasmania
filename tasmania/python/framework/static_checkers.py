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
from typing import Sequence, TYPE_CHECKING

from sympl._core.exceptions import InvalidPropertyDictError
from sympl._core.static_operators import StaticComponentOperator
from sympl._core.units import units_are_compatible

from tasmania.python.framework.exceptions import (
    IncompatibleDimensionsError,
    IncompatibleUnitsError,
)

if TYPE_CHECKING:
    from sympl._core.typingx import PropertyDict

    from tasmania.python.utils.typingx import Component


def check_dims_are_compatible(
    dim1: Sequence[str], dim2: Sequence[str]
) -> None:
    if "*" not in dim1 and "*" not in dim2:
        if len(dim1) != len(dim2):
            raise IncompatibleDimensionsError(dim1, dim2)

        if any(d not in dim2 for d in dim1) or any(
            d not in dim1 for d in dim2
        ):
            raise IncompatibleDimensionsError(dim1, dim2)
    elif "*" in dim1 and "*" not in dim2:
        for d in dim1:
            if d != "*" and d not in dim2:
                raise IncompatibleDimensionsError(dim1, dim2)
    elif "*" not in dim1 and "*" in dim2:
        for d in dim2:
            if d != "*" and d not in dim1:
                raise IncompatibleDimensionsError(dim1, dim2)


def check_units_are_compatible(unit1: str, unit2: str) -> None:
    if not units_are_compatible(unit1, unit2):
        raise IncompatibleUnitsError(unit1, unit2)


def _check_properties_are_compatible(
    properties1: "PropertyDict", properties2: "PropertyDict", units_suffix: str
) -> None:
    shared_keys = set(properties1.keys()).intersection(properties2.keys())
    for key in shared_keys:
        prop1 = properties1[key]
        prop2 = properties2[key]
        if "dims" in prop1 and "dims" in prop2:
            check_dims_are_compatible(prop1["dims"], prop2["dims"])
        if "units" in prop1 and "units" in prop2:
            check_units_are_compatible(
                prop1["units"] + units_suffix, prop2["units"]
            )


def check_properties_are_compatible(
    component: "Component",
    properties_name: str,
    other_component: "Component",
    other_properties_name: str,
    units_suffix: str = "",
) -> None:
    operator = StaticComponentOperator.factory(properties_name)
    properties = operator.get_properties_with_dims(component)
    other_operator = StaticComponentOperator.factory(other_properties_name)
    other_properties = other_operator.get_properties_with_dims(other_component)

    try:
        _check_properties_are_compatible(
            properties, other_properties, units_suffix
        )
    except (IncompatibleDimensionsError, IncompatibleUnitsError) as err:
        name = component.__class__.__name__
        other_name = other_component.__class__.__name__
        if name == other_name:
            raise InvalidPropertyDictError(
                f"{properties_name} and {other_properties_name} of {name} "
                f"are incompatible: {err}."
            )
        else:
            raise InvalidPropertyDictError(
                f"{properties_name} of {name} and {other_properties_name} of "
                f"{other_name} are incompatible: {err}."
            )


def check_missing_fields(
    component: "Component",
    properties_name: str,
    other_component: "Component",
    other_properties_name: str,
) -> None:
    operator = StaticComponentOperator.factory(properties_name)
    properties = operator.get_properties(component)
    other_operator = StaticComponentOperator.factory(other_properties_name)
    other_properties = other_operator.get_properties(other_component)

    missing_fields = set(properties.keys()).difference(other_properties.keys())

    if len(missing_fields):
        other_name = other_component.__class__.__name__
        raise InvalidPropertyDictError(
            f"{other_properties_name} of {other_name} is missing the "
            f"following fields: {', '.join(missing_fields)}."
        )
