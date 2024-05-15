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
from sympl import DataArray
from typing import Dict, Mapping, Optional


def get_constant(
    name: str, units: str, default_value: Optional[DataArray] = None
) -> float:
    """
    Get the value of a physical constant in the desired units.
    The function first looks for the constant in :mod:`tasmania.namelist`.
    If not found, it then searches in `sympl._core.constants.constants`.
    If still not found, the function reverts to `default_value`, which is
    added to :obj:`sympl._core.constants.constants` before returning.

    Parameters
    ----------
    name : str
        Name of the physical constant.
    units : str
        Units in which the constant should be expressed.
    default_value : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the default value for the
        physical constant.

    Return
    ------
    float :
        Value of the physical constant.

    Raises
    ------
    ValueError :
        If the constant cannot be expressed in the desired units.
    ConstantNotFoundError :
        If the constant cannot be found.
    """
    try:
        exec("from tasmania.namelist import {} as var".format(name))
        return locals()["var"].to_units(units).values.item()
    except (ImportError, AttributeError):
        try:
            from sympl import get_constant as sympl_get_constant

            return sympl_get_constant(name, units)
        except KeyError:
            if default_value is not None:
                return_value = default_value.to_units(units).values.item()
                from sympl import set_constant as sympl_set_constant

                sympl_set_constant(name, return_value, units)
                return return_value
            else:
                from tasmania.python.utils.exceptions import (
                    ConstantNotFoundError,
                )

                raise ConstantNotFoundError("{} not found".format(name))


def get_physical_constants(
    default_physical_constants: Mapping[str, DataArray],
    physical_constants: Mapping[str, DataArray] = None,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    default_physical_constants : dict[str, sympl.DataArray]
        Dictionary whose keys are names of some physical constants,
        and whose values are :class:`sympl.DataArray`\s storing the
        default values and units of those constants.
    physical_constants : `dict[str, sympl.DataArray]`, optional
        Dictionary whose keys are names of some physical constants,
        and whose values are :class:`sympl.DataArray`\s storing the
        values and units of those constants.

    Return
    ------
    dict[str, float] :
        Dictionary whose keys are the names of the physical constants
        contained in `default_physical_constants`, and whose values
        are the values of those constants in the default units.
        The function first looks for the value of each constant in
        `physical_constants`. If this is not given, or it does not
        contain that constant, the value is retrieved via
        :func:`tasmania.utils.data_utils.get_constant`, using the corresponding
        value of `default_physical_constants` as default.
    """
    raw_physical_constants = {}

    for name, d_const in default_physical_constants.items():
        d_units = d_const.attrs["units"]
        const = (
            physical_constants.get(name, None)
            if physical_constants is not None
            else None
        )
        raw_physical_constants[name] = (
            const.to_units(d_units).values.item()
            if const is not None
            else get_constant(name, d_units, default_value=d_const)
        )

    return raw_physical_constants
