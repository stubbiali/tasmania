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
	equal_to
	smaller_than
	smaller_or_equal_than
	greater_than
	greater_or_equal_than
	convert_datetime64_to_datetime
	assert_sequence
	to_units
"""
from datetime import datetime
import math
import numpy as np

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

	ts = (time - np.datetime64('0', 's')) / np.timedelta64(1, 's')
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


def to_units(field, units=None):
	return field if units is None else field.to_units(units)
