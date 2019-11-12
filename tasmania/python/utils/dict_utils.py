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
from copy import deepcopy
from sympl import DataArray
from sympl._core.units import units_are_same

from tasmania.python.utils.storage_utils import zeros


def add(
    state_1,
    state_2,
    out=None,
    units=None,
    unshared_variables_in_output=True,
    *,
    backend="numpy",
    default_origin=None
):
    """
    Sum two dictionaries of :class:`sympl.DataArray`\s.

    Parameters
    ----------
    state_1 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    state_2 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables
        included in the output dictionary, and values are strings indicating
        the units in which those variables should be expressed.
        If not specified, a variable is included in the output dictionary
        in the same units used in the first input dictionary, or the second
        dictionary if the variable is not present in the first one.
    unshared_variables_in_output : `bool`, optional
        `True` if the output dictionary should contain those variables
        included in only one of the two input dictionaries, `False` otherwise.
        Defaults to `True`.

    Return
    ------
    dict[str, sympl.DataArray] :
        The sum of the two input dictionaries.
    """
    units = {} if units is None else units

    out_state = {} if out is None else out
    if "time" in state_1 or "time" in state_2:
        out_state["time"] = state_1.get("time", state_2.get("time", None))

    for key in set().union(state_1.keys(), state_2.keys()):
        if key != "time":
            if (state_1.get(key, None) is not None) and (
                state_2.get(key, None) is not None
            ):
                _units = units.get(key, state_2[key].attrs["units"])

                if key in out_state:
                    out_da = out_state[key]
                    out_da.values[...] = state_2[key].to_units(_units).values
                    out_da.attrs["units"] = _units
                elif units_are_same(state_2[key].attrs["units"], _units):
                    out_da = deepcopy(state_2[key])
                else:
                    shape = state_2[key].shape
                    _backend = state_2[key].attrs.get("backend", backend)
                    dtype = state_2[key].dtype
                    _default_origin = state_2[key].attrs.get(
                        "default_origin", default_origin
                    )

                    out_array = zeros(
                        shape, _backend, dtype, default_origin=_default_origin
                    )

                    out_da = DataArray(
                        out_array,
                        dims=state_2[key].dims,
                        coords=state_2[key].coords,
                        attrs={"units": _units},
                    )
                    out_da.values[...] = state_2[key].to_units(_units).values

                out_da.values += state_1[key].to_units(_units).values
                out_state[key] = out_da
            elif unshared_variables_in_output:
                if (state_1.get(key, None) is not None) and (
                    state_2.get(key, None) is None
                ):
                    _units = units.get(key, state_1[key].attrs["units"])
                    if key in out_state:
                        out_state[key].values[...] = state_1[key].to_units(_units)
                        out_state[key].attrs["units"] = _units
                    else:
                        out_state[key] = state_1[key].to_units(_units)
                else:
                    _units = units.get(key, state_2[key].attrs["units"])
                    if key in out_state:
                        out_state[key].values[...] = state_2[key].to_units(_units)
                        out_state[key].attrs["units"] = _units
                    else:
                        out_state[key] = state_2[key].to_units(_units)

    return out_state


def add_inplace(state_1, state_2, units=None, unshared_variables_in_output=True):
    """
    Sum two dictionaries of :class:`sympl.DataArray`\s.

    Parameters
    ----------
    state_1 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    state_2 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables
        included in the output dictionary, and values are strings indicating
        the units in which those variables should be expressed.
        If not specified, a variable is included in the output dictionary
        in the same units used in the first input dictionary, or the second
        dictionary if the variable is not present in the first one.
    unshared_variables_in_output : `bool`, optional
        `True` if the output dictionary should contain those variables
        included in only one of the two input dictionaries, `False` otherwise.
        Defaults to `True`.
    """
    units = {} if units is None else units

    for key in set().union(state_1.keys(), state_2.keys()):
        if key != "time":
            if (state_1.get(key, None) is not None) and (
                state_2.get(key, None) is not None
            ):
                _units = units.get(key, state_1[key].attrs["units"])
                field_1 = state_1[key].to_units(_units).values
                field_2 = state_2[key].to_units(_units).values
                state_1[key].values[...] = field_1 + field_2
                state_1[key].attrs["units"] = _units
            elif unshared_variables_in_output:
                if (state_1.get(key, None) is not None) and (
                    state_2.get(key, None) is None
                ):
                    _units = units.get(key, state_1[key].attrs["units"])
                else:
                    _units = units.get(key, state_2[key].attrs["units"])
                    state_1[key] = state_2[key]

                if state_1[key].attrs["units"] != _units:
                    field_1 = state_1[key].to_units(_units).values
                    state_1[key].attrs["units"] = _units
                    state_1[key].values[...] = field_1


def subtract(
    state_1,
    state_2,
    out=None,
    units=None,
    unshared_variables_in_output=True,
    *,
    backend="numpy",
    default_origin=None
):
    """
    Subtract two dictionaries of :class:`sympl.DataArray`\s.

    Parameters
    ----------
    state_1 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    state_2 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables
        included in the output state, and values are strings indicating
        the units in which those variables should be expressed.
        If not specified, a variable is included in the output dictionary
        in the same units used in the second dictionary, or the first dictionary
        if the variable is not present in the first one.
    unshared_variables_in_output : `bool`, optional
        `True` if the output state should include those variables included
        in only one of the two input dictionaries (unchanged if present	in the
        first dictionary, with opposite sign if present in the second dictionary),
        `False` otherwise. Defaults to `True`.

    Return
    ------
    dict[str, sympl.DataArray] :
        The subtraction of the two input dictionaries.
    """
    units = {} if units is None else units

    out_state = {} if out is None else out
    if "time" in state_1 or "time" in state_2:
        out_state["time"] = state_1.get("time", state_2.get("time", None))

    for key in set().union(state_1.keys(), state_2.keys()):
        if key != "time":
            if (state_1.get(key, None) is not None) and (
                state_2.get(key, None) is not None
            ):
                _units = units.get(key, state_1[key].attrs["units"])

                if key in out_state:
                    out_da = out_state[key]
                    out_da.values[...] = state_1[key].to_units(_units).values
                    out_da.attrs["units"] = _units
                elif units_are_same(state_1[key].attrs["units"], _units):
                    out_da = deepcopy(state_1[key])
                else:
                    shape = state_1[key].shape
                    _backend = state_1[key].attrs.get("backend", backend)
                    dtype = state_1[key].dtype
                    _default_origin = state_1[key].attrs.get(
                        "default_origin", default_origin
                    )

                    out_array = zeros(
                        shape, _backend, dtype, default_origin=_default_origin
                    )

                    units = {} if units is None else units
                    out_da = DataArray(
                        out_array,
                        dims=state_1[key].dims,
                        coords=state_1[key].coords,
                        attrs={"units": _units},
                    )
                    out_da.values[...] = state_1[key].to_units(_units).values

                out_da.values -= state_2[key].to_units(_units).values
                out_state[key] = out_da
            elif unshared_variables_in_output:
                if (state_1.get(key, None) is not None) and (
                    state_2.get(key, None) is None
                ):
                    _units = units.get(key, state_1[key].attrs["units"])
                    if key in out_state:
                        out_state[key].values[...] = state_1[key].to_units(_units)
                        out_state[key].attrs["units"] = _units
                    else:
                        out_state[key] = state_1[key].to_units(_units)
                else:
                    _units = units.get(key, state_2[key].attrs["units"])
                    if key in out_state:
                        out_state[key].values[...] = state_2[key].to_units(_units)
                        out_state[key].attrs["units"] = _units
                    else:
                        out_state[key] = state_2[key].to_units(_units)

    return out_state


def subtract_inplace(state_1, state_2, units=None, unshared_variables_in_output=True):
    """
    Subtract two dictionaries of :class:`sympl.DataArray`\s.

    Parameters
    ----------
    state_1 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    state_2 : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables
        included in the output state, and values are strings indicating
        the units in which those variables should be expressed.
        If not specified, a variable is included in the output dictionary
        in the same units used in the second dictionary, or the first dictionary
        if the variable is not present in the first one.
    unshared_variables_in_output : `bool`, optional
        `True` if the output state should include those variables included
        in only one of the two input dictionaries (unchanged if present	in the
        first dictionary, with opposite sign if present in the second dictionary),
        `False` otherwise. Defaults to `True`.
    """
    units = {} if units is None else units

    for key in set().union(state_1.keys(), state_2.keys()):
        if key != "time":
            if (state_1.get(key, None) is not None) and (
                state_2.get(key, None) is not None
            ):
                _units = units.get(key, state_1[key].attrs["units"])
                field_1 = state_1[key].to_units(_units).values
                field_2 = state_2[key].to_units(_units).values
                state_1[key].values[...] = field_1 - field_2
                state_1[key].attrs["units"] = _units
            elif unshared_variables_in_output:
                if (state_1.get(key, None) is not None) and (
                    state_2.get(key, None) is None
                ):
                    _units = units.get(key, state_1[key].attrs["units"])
                else:
                    _units = units.get(key, state_2[key].attrs["units"])
                    state_1[key] = -state_2[key]

                field_1 = state_1[key].to_units(_units).values
                state_1[key].values[...] = field_1
                state_1[key].attrs["units"] = _units


def multiply(factor, state, out=None, units=None):
    """
    Scale all :class:`sympl.DataArray`\s contained in a dictionary by a scalar factor.

    Parameters
    ----------
    factor : float
        The factor.
    state : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    out : `dict[str, sympl.DataArray]`, optional
        Dictionary of buffers into which the scaled fields are written.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables included in
        the input dictionary, and values are strings indicating the units in
        which those variables should be expressed. If not specified, variables
        are included in the output dictionary in the same units used in the
        input dictionary.

    Return
    ------
    dict[str, sympl.DataArray] :
        The scaled input dictionary.
    """
    units = {} if units is None else units

    out_state = {} if out is None else out
    if "time" in state:
        out_state["time"] = state["time"]

    for key in state.keys():
        if key != "time":
            _units = units.get(key, state[key].attrs.get("units", None))
            assert _units is not None

            if key in out_state:
                field = state[key].to_units(_units).values
                out_state[key].values[...] = factor * field
                out_state[key].attrs["units"] = _units
            else:
                out_state[key] = factor * state[key].to_units(_units)

    return out_state


def multiply_inplace(factor, state, units=None):
    """
    Scale all :class:`sympl.DataArray`\s contained in a dictionary by a scalar factor.

    Parameters
    ----------
    factor : float
        The factor.
    state : dict[str, sympl.DataArray]
        Dictionary whose keys are strings indicating variable names, and values
        are :class:`sympl.DataArray`\s containing the data for those variables.
    units : `dict[str, str]`, optional
        Dictionary whose keys are strings indicating the variables included in
        the input dictionary, and values are strings indicating the units in
        which those variables should be expressed. If not specified, variables
        are included in the output dictionary in the same units used in the
        input dictionary.

    Return
    ------
    dict[str, sympl.DataArray]
        The scaled input dictionary.
    """
    units = {} if units is None else units

    for key in state.keys():
        if key != "time":
            _units = units.get(key, None)
            if _units is not None:
                field = state[key].to_units(_units).values
                state[key].values[...] = factor * field
                state[key].attrs["units"] = _units
            else:
                state[key].values *= factor


def copy(state_1, state_2):
    """
    Overwrite the :class:`sympl.DataArrays` in one dictionary using the
    :class:`sympl.DataArrays` contained in another dictionary.

    Parameters
    ----------
    state_1 : dict[str, sympl.DataArray]
        The destination dictionary.
    state_2 : dict[str, sympl.DataArray]
        The source dictionary.
    """
    if "time" in state_2:
        state_1["time"] = state_2["time"]

    shared_keys = tuple(key for key in state_1 if key in state_2 and key != "time")
    for key in shared_keys:
        state_1[key].values[...] = (
            state_2[key].to_units(state_1[key].attrs["units"]).values
        )
