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
import numpy as np

import utils

class Axis:
	"""
	Class representing a one-dimensional axis. The class API is designed to be similar to 
	that provided by :class:`xarray.DataArray`.

	Attributes
	----------
	coords : list 
		One-dimensional :class:`numpy.ndarray` storing axis coordinates, wrapped within a list.
	values : array_like
		One-dimensional :class:`numpy.ndarray` storing axis coordinates. This attrribute is semantically identical 
		to :attr:`coords` and it is introduced only for the sake of compliancy with :class:`xarray.DataArray`'s API.
	dims : str 
		Axis dimension, i.e., label.
	attrs : dict
		Axis attributes, e.g., the units.
	"""
	def __init__(self, coords, dims, attrs = None):
		"""
		Constructor.

		Parameters
		----------
		coords : array_like
			One-dimensional :class:`numpy.ndarray` representing the axis values.
		dims : str
			Axis label.
		attrs : `dict`, optional
			Axis attributes. This may be used to specify, e.g., the units, which, following the 
			`CF Conventions <http://cfconventions.org>`_, may be either:
				* 'm' (meters) or multiples, for height-based coordinates;
				* 'Pa' (Pascal) or multiples, for pressure-based coordinates;
				* 'K' (Kelvin), for temperature-based coordinates;
				* 'degrees_east', for longitude;
				* 'degrees_north', for latitude.
		"""
		coords = self._check_arguments(coords, attrs)

		self.coords = [coords]
		self.values = coords
		self.dims   = dims
		self.attrs  = dict() if attrs is None else attrs

	def __getitem__(self, i):
		"""
		Get direct access to the coordinate vector.

		Parameters
		----------
		i : `int` or `array_like
			The index, or a sequence of indices.

		Return
		------
		float : 
			The coordinate(s).
		"""
		return self.values[i]

	def _check_arguments(self, coords, attr):
		"""
		Convert user-specified units to base units, e.g., km --> m, hPa --> Pa.

		Parameters
		----------
		coords : array_like
			One-dimensional :class:`numpy.ndarray` representing the axis values.
		attrs : dict
			Axis attributes. This may be used to specify, e.g., the units, which, following the 
			`CF Conventions <http://cfconventions.org>`_, may be either:
				* 'm' (meters) or multiples, for height-based coordinates;
				* 'Pa' (Pascal) or multiples, for pressure-based coordinates;
				* 'K' (Kelvin), for temperature-based coordinates;
				* 'degrees_east', for longitude;
				* 'degrees_north', for latitude.

		Return
		------
		array_like :
			The axis coordinates expressed in base units.
		"""
		if attr is not None:
			units = attr.get('units', None)
			if units not in [None, 'degrees_east', 'degrees_north']:
				factor = utils.get_factor(units)
				coords = factor * coords
				
		return coords

