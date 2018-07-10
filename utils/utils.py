from datetime import datetime
import math
import numpy as np
import os
import shutil

from tasmania.namelist import tol as _tol

def equal_to(a, b, tol=None):
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
		tol = _tol
	return math.fabs(a - b) <= tol

def smaller_than(a, b, tol=None):
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
		tol = _tol
	return a < (b - tol)

def smaller_or_equal_than(a, b, tol=None):
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
		tol = _tol
	return a <= (b + tol)

def greater_than(a, b, tol=None):
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
		tol = _tol
	return a > (b + tol)

def greater_or_equal_than(a, b, tol=None):
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

	ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
	return datetime.utcfromtimestamp(ts)

def get_numpy_arrays(state, indices, *args):
	"""
	Given a dictionary of :class:`xarray.DataArray`\s and a set of keys, extract the corresponding 
	:class:`numpy.ndarray`\s.

	Parameters
	----------
	state : dict
		A dictionary of :class:`xarray.DataArray`\s.
	indices : tuple
		Tuple of indices or slices identifying the portion of each :class:`xarray.DataArray` to be extracted.
	*args : `str` or `tuple of str`
		Each optional positional argument may be either a strings, specifying the variable to extract,
		or a tuple of aliases for the variable.

	Return
	------
	array_like or list:
		The desired :class:`numpy.ndarray`, or a list collecting the desired :class:`numpy.ndarray`\s.

	Raises
	------
	KeyError :
		If a variable which should be extracted is not included in the input dictionary.
	"""
	raw_state = []

	for key in args:
		if type(key) is str: 				# key represents the name of the variable
			if key not in state.keys():
				raise KeyError('Variable {} not included in the input dictionary.'.format(key))
			raw_state.append(state[key].values[indices])
		elif type(key) in [tuple, list]: 	# key represents a set of aliases for the variable
			for i, alias in enumerate(key):
				if state.get(alias) is not None:
					raw_state.append(state[alias].values[indices])
					break
				elif i == len(key)-1:
					raise KeyError('Neither of the aliases {} is included in the input dictionary.'.format(key))
	
	return raw_state if len(raw_state) > 1 else raw_state[0]

def assert_sequence(seq, reflen=None, reftype=None):
	if reflen is not None:
		assert len(seq) == reflen, \
			'The input sequence has length {}, but {} was expected.'.format(len(seq), reflen)

	if reftype is not None:
		if type(reftype) not in (list, tuple):
			reftype = [reftype,]
		for item in seq:
			error_msg = 'An item of the input sequence is of type ' + str(type(item)) \
						+ ', but one of [ '
			for reftype_ in reftype:
				error_msg += str(reftype_) + ' '
			error_msg += '] was expected.'

			assert type(item) in reftype, error_msg
