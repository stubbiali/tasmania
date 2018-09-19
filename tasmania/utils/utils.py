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


def check_property_compatibility(property_1, property_2, name=None):
	from sympl._core.util import combine_dims, units_are_compatible, \
								 InvalidPropertyDictError

	if 'dims' not in property_1.keys() or 'units' not in property_1.keys() or \
	   'dims' not in property_2.keys() or 'units' not in property_2.keys():
		raise InvalidPropertyDictError()

	try:
		_ = combine_dims(property_1['dims'], property_2['dims'])
	except InvalidPropertyDictError:
		raise InvalidPropertyDictError('Incompatibility between dims {} and {} '
									   'of quantity {}.'
									   .format(property_1['dims'], property_2['dims'], name))

	if not units_are_compatible(property_1['units'], property_2['units']):
		raise InvalidPropertyDictError('Incompatibility between units {} and {} '
									   'for quantity {}.'
									   .format(property_1['units'], property_2['units'], name))
