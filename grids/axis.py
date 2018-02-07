import numpy as np

import utils

class Axis:
	"""
	Class representing a one-dimensional axis. The class API has been designed to be similar to 
	that provided by :class:`xarray.DataArray`.

	Attributes:
		coords (list): One-dimensional :class:`numpy.ndarray` storing axis coordinates, wrapped
			within a list.
		values (array_like): One-dimensional :class:`numpy.ndarray` storing axis coordinates.
			This attrribute is semantically identical to :attr:`coords` and it is
			introduced just for the sake of compliancy with :class:`xarray.DataArray`'s API.
		dims (list): Axis dimension, i.e., label.
		attrs (dict): Axis attributes, e.g., the units.
	"""
	def __init__(self, coords, dims, attrs = None):
		"""
		Constructor.

		Args:
			coords (array_like): Axis coordinates. Must be a one-dimensional :class:`numpy.ndarray`.
			dims (str): Axis label.
			attrs (`dict`, optional): Axis attributes. This may be used to specify, e.g., the units, 
				which, following the `CF Conventions <http://cfconventions.org>`_, may be either:
					* 'm' (meters) or multiples, for height-based coordinates;
					* 'Pa' (Pascal) or multiples, for pressure-based coordinates;
					* 'K' (Kelvin), for temperature-based coordinates;
					* 'degrees_east', for longitude; or
					* 'degrees_north', for latitude.
		"""
		coords = self._check_arguments(coords, attrs)

		self.coords = [coords]
		self.values = coords
		self.dims   = dims
		self.attrs  = dict() if attrs is None else attrs

	def __getitem__(self, i):
		"""
		Get directly access to the coordinate vector.

		Arguments:
			i (int or array_like): The index, or a sequence of indices.

		Return:
			float: The coordinate(s).
		"""
		return self.values[i]

	def _check_arguments(self, coords, attr):
		# Convert user-specified units to base units, e.g.
		# km  --> m
		# hPa --> Pa
		if attr is not None:
			units = attr.get('units', None)
			if units not in [None, 'degrees_east', 'degrees_north']:
				factor = utils.get_factor(units)
				coords = factor * coords
				
		return coords

