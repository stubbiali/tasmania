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
"""
This module contains:
	add
	subtract
	multiply
	copy
"""
def add(state_1, state_2, units=None, unshared_variables_in_output=True):
	"""
	Sum two dictionaries of :class:`sympl.DataArray`\s.

	Parameters
	----------
	state_1 : dict
		Dictionary whose keys are strings indicating variable names, and values
		are :class:`sympl.DataArray`\s containing the data for those variables.
	state_2 : dict
		Dictionary whose keys are strings indicating variable names, and values
		are :class:`sympl.DataArray`\s containing the data for those variables.
	units : `dict`, optional
		Dictionary whose keys are strings indicating the variables
		included in the output dictionary, and values are strings indicating
		the units in which those variables should be expressed.
		If not specified, a variable is included in the output dictionary
		in the same units used in the first input dictionary, or the second
		dictionary if the variable is not present in the first one.
	unshared_variables_in_output : `bool`, optional
		:obj:`True` if the output dictionary should contain those variables
		included in only one of the two input dictionaries, :obj:`False` otherwise.
		Defaults to :obj:`True`.

	Return
	------
	dict :
		The sum of the two input dictionaries.
	"""
	units = {} if units is None else units

	try:
		out_state = {'time': state_1['time']}
	except KeyError:
		out_state = {}

	for key in set().union(state_1.keys(), state_2.keys()):
		if key != 'time':
			if (state_1.get(key, None) is not None) and (state_2.get(key, None) is not None):
				if units.get(key, None) is not None:
					out_state[key] = state_1[key].to_units(units[key]) + \
									 state_2[key].to_units(units[key])
				else:
					out_state[key] = state_1[key] + \
									 state_2[key].to_units(state_1[key].attrs['units'])

			if unshared_variables_in_output:
				if (state_1.get(key, None) is not None) and (state_2.get(key, None) is None):
					if units.get(key, None) is not None:
						out_state[key] = state_1[key].to_units(units[key])
					else:
						out_state[key] = state_1[key]

				if (state_1.get(key, None) is None) and (state_2.get(key, None) is not None):
					if units.get(key, None) is not None:
						out_state[key] = state_2[key].to_units(units[key])
					else:
						out_state[key] = state_2[key]

	return out_state


def subtract(state_1, state_2, units=None, unshared_variables_in_output=True):
	"""
	Subtract two dictionaries of :class:`sympl.DataArray`\s.

	Parameters
	----------
	state_1 : dict
		Dictionary whose keys are strings indicating variable names, and values
		are :class:`sympl.DataArray`\s containing the data for those variables.
	state_2 : dict
		Dictionary whose keys are strings indicating variable names, and values
		are :class:`sympl.DataArray`\s containing the data for those variables.
	units : `dict`, optional
		Dictionary whose keys are strings indicating the variables
		included in the output state, and values are strings indicating
		the units in which those variables should be expressed.
		If not specified, a variable is included in the output dictionary
		in the same units used in the first dictionary, or the second dictionary
		if the variable is not present in the first one.
	unshared_variables_in_output : `bool`, optional
		:obj:`True` if the output state should include those variables included 
		in only one of the two input dictionaries (unchanged if present	in the 
		first dictionary, with opposite sign if present in the second dictionary),
		:obj:`False` otherwise. Defaults to :obj:`True`.

	Return
	------
	dict :
		The subtraction of the two input dictionaries.
	"""
	units = {} if units is None else units

	try:
		out_state = {'time': state_1['time']}
	except KeyError:
		out_state = {}

	for key in set().union(state_1.keys(), state_2.keys()):
		if key != 'time':
			if (state_1.get(key, None) is not None) and (state_2.get(key, None) is not None):
				if units.get(key, None) is not None:
					out_state[key] = state_1[key].to_units(units[key]) - \
									 state_2[key].to_units(units[key])
				else:
					out_state[key] = state_1[key] - \
									 state_2[key].to_units(state_1[key].attrs['units'])

			if unshared_variables_in_output:
				if (state_1.get(key, None) is not None) and (state_2.get(key, None) is None):
					if units.get(key, None) is not None:
						out_state[key] = state_1[key].to_units(units[key])
					else:
						out_state[key] = state_1[key]

				if (state_1.get(key, None) is None) and (state_2.get(key, None) is not None):
					if units.get(key, None) is not None:
						out_state[key] = - state_2[key].to_units(units[key])
					else:
						out_state[key] = - state_2[key]

	return out_state


def multiply(factor, state, units=None):
	"""
	Scale all :class:`sympl.DataArray`\s contained in a dictionary by a scalar factor.

	Parameters
	----------
	factor : float
		The factor.
	state : dict
		Dictionary whose keys are strings indicating variable names, and values
		are :class:`sympl.DataArray`\s containing the data for those variables.
	units : `dict`, optional
		Dictionary whose keys are strings indicating the variables included in 
		the input dictionary, and values are strings indicating the units in 
		which those variables should be expressed. If not specified, variables 
		are included in the output dictionary in the same units used in the 
		input dictionary.

	Return
	------
	dict :
		The scaled input dictionary.
	"""
	units = {} if units is None else units

	try:
		out_state = {'time': state['time']}
	except KeyError:
		out_state = {}

	for key in state.keys():
		if key != 'time':
			if units.get(key, None) is not None:
				val = state[key].to_units(units[key])
				out_state[key] = factor * val
				out_state[key].attrs.update(val.attrs)
			else:
				out_state[key] = factor * state[key]
				out_state[key].attrs.update(state[key].attrs)

	return out_state


def copy(state_1, state_2):
	"""
	Overwrite the :class:`sympl.DataArrays` in one dictionary using the
	:class:`sympl.DataArrays` contained in another dictionary.

	Parameters
	----------
	state_1 : dict
		The destination dictionary.
	state_2 : dict
		The source dictionary.
	"""
	shared_keys = tuple(key for key in state_1 if key in state_2)
	for key in shared_keys:
		if key != 'time':
			state_1[key][...] = state_2[key].to_units(state_1[key].attrs['units'])[...]