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
from datetime import datetime
import math
import numpy as np

from sympl._core.combine_properties import \
	combine_dims, units_are_compatible, InvalidPropertyDictError
from sympl._core.units import clean_units

try:
	from tasmania.namelist import tol as d_tol
except ImportError:
	d_tol = 1e-8


def equal_to(a, b, tol=d_tol):
	"""
	Compare floating point numbers, or arrays of floating point numbers,
	properly accounting for round-off errors.

	Parameters
	----------
	a : `float` or `array_like` 
		Left-hand side.
	b : `float` or `array_like` 
		Right-hand side.
	tol : `float`, optional 
		Tolerance.

	Return
	------
	bool : 
		:obj:`True` if :data:`a` is equal to :data:`b` up to :data:`tol`,
		:obj:`False` otherwise.
	"""
	return math.fabs(a - b) <= tol


def smaller_than(a, b, tol=d_tol):
	"""
	Compare floating point numbers, or arrays of floating point numbers,
	properly accounting for round-off errors.

	Parameters
	----------
	a : `float` or `array_like` 
		Left-hand side.
	b : `float` or `array_like` 
		Right-hand side.
	tol : `float`, optional 
		Tolerance.

	Return
	------
	bool : 
		:obj:`True` if :data:`a` is smaller than :data:`b` up to :data:`tol`,
		:obj:`False` otherwise.
	"""
	return a < (b - tol)


def smaller_or_equal_than(a, b, tol=d_tol):
	"""
	Compare floating point numbers or arrays of floating point numbers,
	properly accounting for round-off errors.

	Parameters
	----------
	a : `float` or `array_like` 
		Left-hand side.
	b : `float` or `array_like` 
		Right-hand side.
	tol : `float`, optional 
		Tolerance.

	Return
	------
	bool : 
		:obj:`True` if :data:`a` is smaller than or equal to :data:`b`
		up to :data:`tol`, :obj:`False` otherwise.
	"""
	return a <= (b + tol)


def greater_than(a, b, tol=d_tol):
	"""
	Compare floating point numbers, or arrays of floating point numbers,
	properly accounting for round-off errors.

	Parameters
	----------
	a : `float` or `array_like` 
		Left-hand side.
	b : `float` or `array_like` 
		Right-hand side.
	tol : `float`, optional 
		Tolerance.

	Return
	------
	bool :  
		:obj:`True` if :data:`a` is greater than :data:`b` up to :data:`tol`,
		:obj:`False` otherwise.
	"""
	return a > (b + tol)


def greater_or_equal_than(a, b, tol=d_tol):
	"""
	Compare floating point numbers, or arrays of floating point numbers,
	properly accounting for round-off errors.

	Parameters
	----------
	a : `float` or `array_like` 
		Left-hand side.
	b : `float` or `array_like` 
		Right-hand side.
	tol : `float`, optional 
		Tolerance.

	Return
	------
	bool : 
		:obj:`True` if :data:`a` is greater than or equal to :data:`b`
		up to :data:`tol`, :obj:`False` otherwise.
	"""
	return a >= (b - tol)


def convert_datetime64_to_datetime(time):
	"""
	Convert :class:`numpy.datetime64` to :class:`datetime.datetime`.

	Parameters
	----------
	time : obj 
		The :class:`numpy.datetime64` object to convert.

	Return
	------
	obj :
		The converted :class:`datetime.datetime` object.

	References
	----------
	https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64.
	https://github.com/bokeh/bokeh/pull/6192/commits/48aea137edbabe731fb9a9c160ff4ab2b463e036.
	"""
	# Safeguard check
	if type(time) == datetime:
		return time

	ts = (time - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')
	return datetime.utcfromtimestamp(ts)


def assert_sequence(seq, reflen=None, reftype=None):
	if reflen is not None:
		assert len(seq) == reflen, \
			'The input sequence has length {}, but {} was expected.' \
			.format(len(seq), reflen)

	if reftype is not None:
		if type(reftype) is not tuple:
			reftype = (reftype, )
		for item in seq:
			error_msg = 'An item of the input sequence is of type ' \
						+ str(type(item)) + ', but one of [ '
			for reftype_ in reftype:
				error_msg += str(reftype_) + ' '
			error_msg += '] was expected.'

			assert isinstance(item, reftype), error_msg


def check_properties_compatibility(
	properties1, properties2, to_append=None, properties1_name=None, properties2_name=None):
	_properties1 = {}
	if to_append is None:
		_properties1.update(properties1)
	else:
		_properties1.update({
			name: {'dims': value['dims'], 'units': clean_units(value['dims'] + to_append)}
			for name, value in properties1.items()
		})

	shared_vars = set(_properties1.keys()).intersection(properties2.keys())
	for name in shared_vars:
		check_property_compatibility(
			properties1[name], properties2[name], property_name=name,
			origin1_name=properties1_name, origin2_name=properties2_name,
		)


def check_property_compatibility(
	property1, property2, property_name=None, origin1_name=None, origin2_name=None
):
	if 'dims' not in property1.keys() or 'units' not in property1.keys() or \
	   'dims' not in property2.keys() or 'units' not in property2.keys():
		raise InvalidPropertyDictError()

	try:
		_ = combine_dims(property1['dims'], property2['dims'])
	except InvalidPropertyDictError:
		raise InvalidPropertyDictError(
			'Incompatibility between dims {} (in {}) and {} (in {}) of quantity {}.'
			.format(
				property1['dims'],
				'properties1' if origin1_name is None else origin1_name,
				property2['dims'],
				'properties2' if origin2_name is None else origin2_name,
				'unknown' if property_name is None else property_name,
			)
		)

	if not units_are_compatible(property1['units'], property2['units']):
		raise InvalidPropertyDictError(
			'Incompatibility between units {} (in {}) and {} (in {}) of quantity {}.'
			.format(
				property1['units'],
				'properties1' if origin1_name is None else origin1_name,
				property2['units'],
				'properties2' if origin2_name is None else origin2_name,
				'unknown' if property_name is None else property_name,
			)
		)


def check_missing_properties(
	properties1, properties2, properties1_name=None, properties2_name=None
):
	missing_vars = set(properties1.keys()).difference(properties2.keys())

	if len(missing_vars) > 0:
		raise InvalidPropertyDictError(
			'{} are present in {} but missing in {}.'
			.format(
				', '.join(missing_vars),
				'properties1' if properties1_name is None else properties1_name,
				'properties2' if properties2_name is None else properties2_name,
			)
		)


def resolve_aliases(data_dict, properties_dict):
	name_to_alias = _get_name_to_alias_map(data_dict, properties_dict)
	return _replace_aliases(data_dict, name_to_alias)


def _get_name_to_alias_map(data_dict, properties_dict):
	return_dict = {}

	for name in properties_dict:
		aliases = [name, ]
		if properties_dict[name].get('alias', None) is not None:
			aliases.append(properties_dict[name]['alias'])

		for alias in aliases:
			if alias in data_dict:
				return_dict[name] = alias
				break
			else:
				pass

		assert name in return_dict

	return return_dict


def _replace_aliases(data_dict, name_to_alias):
	return_dict = {}

	for name in name_to_alias:
		if name != name_to_alias[name]:
			return_dict[name] = data_dict[name_to_alias[name]]

	return return_dict


def get_time_string(seconds):
	s = ''

	hours = int(seconds / (60*60))
	s += '0{}:'.format(hours) if hours < 10 else '{}:'.format(hours)
	remainder = seconds - hours*60*60

	minutes = int(remainder / 60)
	s += '0{}:'.format(minutes) if minutes < 10 else '{}:'.format(minutes)
	remainder -= minutes*60

	s += '0{}'.format(int(remainder)) if int(remainder) < 10 \
		else '{}'.format(int(remainder))

	return s
