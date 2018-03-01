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
Some useful utilities. 
"""
from datetime import datetime
import math
import numpy as np
import os
import shutil

def equal_to(a, b, tol = None):
	"""
	Compare floating point numbers (or arrays of floating point numbers), properly accounting for round-off errors.

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
		:obj:`True` if :data:`a` is equal to :data:`b` up to :data:`tol`, :obj:`False` otherwise.
	"""
	if tol is None:
		from namelist import tol as _tol
		tol = _tol
	return math.fabs(a - b) <= tol

def smaller_than(a, b, tol = None):
	"""
	Compare floating point numbers (or arrays of floating point numbers), properly accounting for round-off errors.

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
		:obj:`True` if :data:`a` is smaller than :data:`b` up to :data:`tol`, :obj:`False` otherwise.
	"""
	if tol is None:
		from namelist import tol as _tol
		tol = _tol
	return a < (b - tol)

def smaller_or_equal_than(a, b, tol = None):
	"""
	Compare floating point numbers (or arrays of floating point numbers), properly accounting for round-off errors.

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
		:obj:`True` if :data:`a` is smaller than or equal to :data:`b` up to :data:`tol`, :obj:`False` otherwise.
	"""
	if tol is None:
		from namelist import tol as _tol
		tol = _tol
	return a <= (b + tol)

def greater_than(a, b, tol = None):
	"""
	Compare floating point numbers (or arrays of floating point numbers), properly accounting for round-off errors.

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
		:obj:`True` if :data:`a` is greater than :data:`b` up to :data:`tol`, :obj:`False` otherwise.
	"""
	if tol is None:
		from namelist import tol as _tol
		tol = _tol
	return a > (b + tol)

def greater_or_equal_than(a, b, tol = None):
	"""
	Compare floating point numbers (or arrays of floating point numbers), properly accounting for round-off errors.

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
		:obj:`True` if :data:`a` is greater than or equal to :data:`b` up to :data:`tol`, :obj:`False` otherwise.
	"""
	if tol is None:
		from namelist import tol as _tol
		tol = _tol
	return a >= (b - tol)

def get_factor(units):
	"""
	Convert units prefix to the corresponding factor. 
	For the conversion, the `CF Conventions <http://cfconventions.org/>`_ are used.

	Parameters
	----------
	units : str 
		The units.

	Return
	------
	float : 
		The factor.
	"""
	if units == "m":
		return 1.

	prefix_to_factor = {"y" : 1.e-24,
						"z" : 1.e-21,
						"a" : 1.e-18,
						"f" : 1.e-15,
						"p" : 1.e-12,
						"n" : 1.e-9,
						"u" : 1.e-6,
						"m" : 1.e-3,
						"c" : 1.e-2,
						"d" : 1.e-1,
						"da": 1.e1,
						"h" : 1.e2,
						"k" : 1.e3,
						"M" : 1.e6,
						"G" : 1.e9,
						"T" : 1.e12,
						"P" : 1.e15,
						"E" : 1.e18,
						"Z" : 1.e21,
						"Y" : 1.e24}	
	prefix = _get_prefix(units)
	return prefix_to_factor.get(prefix, 1.)

def _get_prefix(units):
	"""
	Extract the prefix from the units name.

	Parameters
	----------
	units : str 
		The units.

	Return
	------
	str : 
		The prefix.
	"""
	if units[:2] == "da":
		prefix = units[:2]
	else:
		prefix = units[0]
	return prefix
		
def set_namelist(user_namelist = None):
	"""
	Place the user-defined namelist module in the Python search path.
	This is achieved by physically copying the content of the user-provided module into TASMANIA_ROOT/namelist.py.

	Parameters
	----------
	user_namelist : str 
		Path to the user-defined namelist. If not specified, the default namelist TASMANIA_ROOT/_namelist.py is used.
	"""
	try:
		tasmania_root = os.environ['TASMANIA_ROOT']
	except RuntimeError:
		print('Hint: has the environmental variable TASMANIA_ROOT been set?')
		raise

	if user_namelist is None: # Default case
		src_file = os.path.join(tasmania_root, '_namelist.py')
		dst_file = os.path.join(tasmania_root, 'namelist.py')
		shutil.copy(src_file, dst_file)
	else:
		src_dir = os.curdir
		src_file = os.path.join(src_dir, user_namelist)
		dst_file = os.path.join(tasmania_root, 'namelist.py')
		shutil.copy(src_file, dst_file)
										
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
	"""
	# Safeguard check
	if type(time) == datetime:
		return time

	ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
	return datetime.utcfromtimestamp(ts)

