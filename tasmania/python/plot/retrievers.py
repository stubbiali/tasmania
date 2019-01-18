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
from sympl._core.units import clean_units

from tasmania.python.grids.grid_xyz import GridXYZ as GridType
from tasmania.python.plot.utils import to_units


class DataRetriever:
	"""
	Callable class retrieving a raw scalar field from a state dictionary.
	"""
	def __init__(self, grid, field_name, field_units=None,
				 x=None, y=None, z=None):
		"""
		Parameters
		----------
		grid : grid
			TODO
		field_name : str
			TODO
		field_units : `str`, optional
			TODO
		x : `slice`, optional
			TODO
		y : `slice`, optional
			TODO
		z : `slice`, optional
			TODO
		"""
		self._grid = grid
		self._fname = field_name
		self._funits = field_units
		self.x = x if x is not None else slice(0, None)
		self.y = y if y is not None else slice(0, None)
		self.z = z if z is not None else slice(0, None)

	def __call__(self, state):
		"""
		Call operator retrieving the field.

		Parameters
		----------
		state : dict
			State dictionary.

		Returns
		-------
		array_like :
			Raw field.
		"""
		grid = self._grid
		field_name, field_units = self._fname, self._funits
		x, y, z = self.x, self.y, self.z

		if field_name in state:  # model variable

			return to_units(state[field_name][x, y, z], field_units).values

		elif field_name == 'horizontal_velocity':  # horizontal velocity magnitude

			try:
				funits = clean_units('kg m^-3' + field_units) if field_units is not None \
						 else 'kg m^-2 s^-1'
				r  = state['air_density'][x, y, z].to_units('kg m^-3').values
				ru = state['x_momentum'][x, y, z].to_units(funits).values
				rv = state['y_momentum'][x, y, z].to_units(funits).values
				u, v = ru / r, rv / r
			except KeyError:
				try:
					funits = clean_units('kg m^-2 K^-1' + field_units) if field_units is not None \
							 else 'kg m^-1 K^-1 s^-1'
					s  = state['air_isentropic_density'][x, y, z].to_units('kg m^-2 K^-1').values
					su = state['x_momentum_isentropic'][x, y, z].to_units(funits).values
					sv = state['y_momentum_isentropic'][x, y, z].to_units(funits).values
					u, v = su / s, sv / s
				except KeyError:
					try:
						u = to_units(state['x_velocity'][x, y, z], field_units).values
						v = to_units(state['y_velocity'][x, y, z], field_units).values
					except KeyError:
						try:
							x_ = slice(x.start, x.stop+1 if x.stop is not None else x.stop)
							y_ = slice(y.start, y.stop+1 if y.stop is not None else y.stop)
							u = to_units(state['x_velocity_at_u_locations'][x_, y, z], field_units).values
							v = to_units(state['y_velocity_at_v_locations'][x, y_, z], field_units).values
							u = 0.5 * (u[:-1, :, :] + u[1:, :, :])
							v = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
						except KeyError:
							raise RuntimeError('Sorry, don\'t know how to retrieve '
											   '\'horizontal_velocity\'.')

			return np.sqrt(u**2 + v**2)

		elif field_name == 'x_velocity':  # unstaggered x-velocity

			try:
				funits = clean_units('kg m^-3' + field_units) if field_units is not None \
					else 'kg m^-2 s^-1'
				r  = state['air_density'][x, y, z].to_units('kg m^-3').values
				ru = state['x_momentum'][x, y, z].to_units(funits).values
				return ru / r
			except KeyError:
				try:
					funits = clean_units('kg m^-2 K^-1' + field_units) if field_units is not None \
						else 'kg m^-1 K^-1 s^-1'
					s  = state['air_isentropic_density'][x, y, z].to_units('kg m^-2 K^-1').values
					su = state['x_momentum_isentropic'][x, y, z].to_units(funits).values
					return su / s
				except KeyError:
					try:
						x_ = slice(x.start, x.stop+1 if x.stop is not None else x.stop)
						u = to_units(state['x_velocity_at_u_locations'][x_, y, z], field_units).values
						return 0.5 * (u[:-1, :, :] + u[1:, :, :])
					except KeyError:
						raise RuntimeError('Sorry, don\'t know how to retrieve \'x_velocity\'.')

		elif field_name == 'y_velocity':  # unstaggered y-velocity

			try:
				funits = clean_units('kg m^-3' + field_units) if field_units is not None \
					else 'kg m^-2 s^-1'
				r  = state['air_density'][x, y, z].to_units('kg m^-3').values
				rv = state['y_momentum'][x, y, z].to_units(funits).values
				return rv / r
			except KeyError:
				try:
					funits = clean_units('kg m^-2 K^-1' + field_units) if field_units is not None \
						else 'kg m^-1 K^-1 s^-1'
					s  = state['air_isentropic_density'][x, y, z].to_units('kg m^-2 K^-1').values
					sv = state['y_momentum_isentropic'][x, y, z].to_units(funits).values
					return sv / s
				except KeyError:
					try:
						y_ = slice(y.start, y.stop+1 if y.stop is not None else y.stop)
						v = to_units(state['y_velocity_at_v_locations'][x, y_, z], field_units).values
						return 0.5 * (v[:, :-1, :] + v[:, 1:, :])
					except KeyError:
						raise RuntimeError('Sorry, don\'t know how to retrieve \'y_velocity\'.')

		elif field_name == 'height':  # geometric height

			try:
				return to_units(grid.height[x, y, z], field_units).values
			except AttributeError:
				z_ = slice(z.start, z.stop+1 if z.stop is not None else z.stop)
				try:
					tmp = to_units(state['height_on_interface_levels'][x, y, z_], field_units).values
					return 0.5 * (tmp[:, :, :-1] + tmp[:, :, 1:])
				except KeyError:
					try:
						tmp = to_units(grid.height_on_interface_levels[x, y, z_], field_units).values
						return 0.5 * (tmp[:, :, :-1] + tmp[:, :, 1:])
					except AttributeError:
						pass

		elif field_name == 'height_on_interface_levels':  # geometric height

			try:
				return to_units(grid.height_on_interface_levels[x, y, z], field_units).values
			except AttributeError:
				pass

		elif field_name == 'air_pressure':  # pressure

			try:
				z_ = slice(z.start, z.stop+1 if z.stop is not None else z.stop)
				tmp = to_units(state['air_pressure_on_interface_levels'][x, y, z_],
							   field_units).values
				return 0.5 * (tmp[:, :, :-1] + tmp[:, :, 1:])
			except KeyError:
				pass

		elif field_name == 'air_pressure_on_interface_levels':  # pressure

			try:
				return to_units(state['air_pressure'][x, y, z], field_units).values
			except KeyError:
				pass

		elif field_name == 'topography':  # topography

			return to_units(grid.topography.topo[x, y], field_units).values

		else:

			raise RuntimeError('Sorry, don\'t know how to retrieve \'{}\'.'.format(field_name))


class DataRetrieverComposite:
	"""
	Callable class retrieving multiple raw fields from multiple states.
	"""
	def __init__(self, grid, field_name, field_units=None, x=None, y=None, z=None):
		"""
		Parameters
		----------
		grid : grid, sequence[grid]
			TODO
		field_name : str, sequence[str], sequence[sequence[str]]
			TODO
		field_units : `str, sequence[str], sequence[sequence[str]]`, optional
			TODO
		x : `slice, sequence[slice], sequence[sequence[slice]]`, optional
			TODO
		y : `slice, sequence[slice], sequence[sequence[slice]]`, optional
			TODO
		z : `slice, sequence[slice], sequence[sequence[slice]]`, optional
			TODO
		"""
		SequenceType = (tuple, list)

		if isinstance(field_name, str):
			fnames = ((field_name, ), )
		elif isinstance(field_name, SequenceType) and \
			all(isinstance(name, str) for name in field_name):
			fnames = (field_name, )
		elif isinstance(field_name, SequenceType) and \
			all(isinstance(name, SequenceType) for name in field_name):
			fnames = field_name
		else:
			raise TypeError('field_name''s type: expected str, sequence[str], or '
							'sequence[sequence[str]], got {}.'.format(type(field_name)))

		if isinstance(grid, GridType):
			grids = (grid, ) * len(fnames)
		elif isinstance(grid, SequenceType) and all(isinstance(g, GridType) for g in grid):
			grids = grid
		else:
			raise TypeError('grid''s type: expected {0}, or sequence[{0}], got {1}.'.format(
				GridType.__class__, type(grid)))

		assert len(grids) == len(fnames), \
			'grid''s length: expected {}, got {}.'.format(len(fnames), len(grids))

		if field_units is None:
			funits = tuple((None, ) * len(arg) for arg in fnames)
		elif isinstance(field_units, str):
			funits = tuple((field_units, ) * len(arg) for arg in fnames)
		elif isinstance(field_units, SequenceType) and \
			all(isinstance(unit, str) for unit in field_units):
			funits = (field_units, )
		elif isinstance(field_units, SequenceType) and \
			all(isinstance(unit, SequenceType) for unit in field_units):
			funits = field_units
		else:
			raise TypeError('field_units''s type: expected str, sequence[str], or '
							'sequence[sequence[str]], got {}.'.format(type(field_units)))

		assert len(funits) == len(fnames), \
			'field_units''s length: expected {}, got{}.'.format(len(fnames), len(funits))
		for i in range(len(funits)):
			assert len(funits[i]) == len(fnames[i]), \
				'field_units[{}]''s length: expected {}, got{}.'.format(
					i, len(fnames[i]), len(funits[i]))

		if x is None:
			fx = tuple((None, ) * len(arg) for arg in fnames)
		elif isinstance(x, slice):
			fx = tuple((x, ) * len(arg) for arg in fnames)
		elif isinstance(x, SequenceType) and \
			all(isinstance(arg, slice) for arg in x):
			fx = (x, )
		elif isinstance(x, SequenceType) and \
			all(isinstance(arg, SequenceType) for arg in x):
			fx = x
		else:
			raise TypeError('x''s type: expected slice, sequence[slice], or '
							'sequence[sequence[slice]], got {}.'.format(type(x)))

		assert len(fx) == len(fnames), \
			'x''s length: expected {}, got{}.'.format(len(fnames), len(fx))
		for i in range(len(fx)):
			assert len(fx[i]) == len(fnames[i]), \
				'x[{}]''s length: expected {}, got{}.'.format(
					i, len(fnames[i]), len(fx[i]))

		if y is None:
			fy = tuple((None, ) * len(arg) for arg in fnames)
		elif isinstance(y, slice):
			fy = tuple((y, ) * len(arg) for arg in fnames)
		elif isinstance(y, SequenceType) and \
			all(isinstance(arg, slice) for arg in y):
			fy = (y, )
		elif isinstance(y, SequenceType) and \
			all(isinstance(arg, SequenceType) for arg in y):
			fy = y
		else:
			raise TypeError('y''s type: expected slice, sequence[slice], or '
							'sequence[sequence[slice]], got {}.'.format(type(y)))

		assert len(fy) == len(fnames), \
			'y''s length: expected {}, got{}.'.format(len(fnames), len(fy))
		for i in range(len(fy)):
			assert len(fy[i]) == len(fnames[i]), \
				'y[{}]''s length: expected {}, got{}.'.format(
					i, len(fnames[i]), len(fy[i]))

		if z is None:
			fz = tuple((None, ) * len(arg) for arg in fnames)
		elif isinstance(z, slice):
			fz = tuple((z, ) * len(arg) for arg in fnames)
		elif isinstance(z, SequenceType) and \
			all(isinstance(arg, slice) for arg in z):
			fz = (z, )
		elif isinstance(z, SequenceType) and \
			all(isinstance(arg, SequenceType) for arg in z):
			fz = z
		else:
			raise TypeError('z''s type: expected slice, sequence[slice], or '
							'sequence[sequence[slice]], got {}.'.format(type(z)))

		assert len(fz) == len(fnames), \
			'z''s length: expected {}, got{}.'.format(len(fnames), len(fz))
		for i in range(len(fy)):
			assert len(fz[i]) == len(fnames[i]), \
				'z[{}]''s length: expected {}, got{}.'.format(
					i, len(fnames[i]), len(fz[i]))

		self._retrievers = []
		for i in range(len(fnames)):
			iretrievers = []
			for j in range(len(fnames[i])):
				iretrievers.append(DataRetriever(grids[i], fnames[i][j], funits[i][j],
												fx[i][j], fy[i][j], fz[i][j]))
			self._retrievers.append(iretrievers)

	def __call__(self, state):
		"""
		TODO

		Parameters
		----------
		state : dict, sequence[dict]
			TODO

		Returns
		-------
		float or array_like :
			TODO
		"""
		args = (state, ) if isinstance(state, dict) else state

		got, expected = len(args), len(self._retrievers)
		if got != expected:
			raise RuntimeError('Expected {} input states, got {}.'.format(expected, got))

		return_seq = []
		for state, state_retrievers in zip(args, self._retrievers):
			local_seq = []
			for retriever in state_retrievers:
				local_seq.append(retriever(state))
			return_seq.append(local_seq)

		return return_seq
