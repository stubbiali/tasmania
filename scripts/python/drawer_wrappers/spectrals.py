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
from .base_wrapper import DrawerWrapper
import json
import numpy as np
import tasmania as taz
from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.plot_utils import make_contourf


class CDFWrapper(DrawerWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader)

		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			field_name = data['field_name']
			field_units = data['field_units']
			x = slice(data['x'][0], data['x'][1], data['x'][2])
			y = slice(data['y'][0], data['y'][1], data['y'][2])
			z = slice(data['z'][0], data['z'][1], data['z'][2])
			drawer_properties = data['drawer_properties']

			self._core = taz.CDF(
				loader.get_grid(), field_name, field_units,
				x=x, y=y, z=z, properties=drawer_properties
			)

	def get_state(self, tlevel):
		tlevel = self._loader.get_nt() + tlevel if tlevel < 0 else tlevel
		drawer_tlevel = -1 if self._core._data is None else len(self._core._data)-1

		if drawer_tlevel >= tlevel:
			for k in range(tlevel, drawer_tlevel+1):
				self._core._data.pop(k)
		else:
			for k in range(drawer_tlevel+1, tlevel):
				self._core(self._loader.get_state(k))

		return self._loader.get_state(tlevel)

  
class StabilityFunction(Drawer):
	def __init__(self, coupling_method, xlim, nx, ylim, ny, properties=None):
		super().__init__(properties)

		xv = np.linspace(xlim[0], xlim[1], nx)
		yv = np.linspace(ylim[0], ylim[1], ny)
		self.x, self.y = np.meshgrid(xv, yv)
		self.cm = coupling_method

	def __call__(self, state, fig, ax):
		cm, x, y = self.cm, self.x, self.y

		if cm == 'fc':
			e = 1 + (x + 1j*y) + 0.5*(x + 1j*y)**2 + 1.0/6.0*(x + 1j*y)**3
		elif cm == 'lfc':
			e = 1 + (x + 1j*y) + 0.5*1j*y*(x + 1j*y) + 1.0/6.0*(x + 1j*y)*(1j*y)**2
		elif cm == 'ps':
			e = 1 + (x + 1j*y) + 0.5*(x**2 + (1j*y)**2) + 1.0/6.0*(1j*y)**3
		elif cm == 'sts':
			e = 1 + (x + 1j*y) + 0.5*((1j*y)**2 + x*1j*y + x**2) + \
				1.0/6.0*((1j*y)**3 + 3.0/2.0*x*(1j*y)**2) + 1.0/12.0*(x*(1j*y)**3)
		elif cm == 'sus':
			e = 1 + (x + 1j*y) + 0.5*(x + 1j*y)**2 + 1.0/6.0*((x + 1j*y)**3 - x**3) \
				+ 1.0/6.0*x*(1j*y)**3 + 0.25*(x**2)*((1j*y)**2) + 1.0/12.0*(x**2)*((1j*y)**3)
		else:
			e = 1 + (x + 1j*y) + 0.5*(x + 1j*y)**2 + 1.0/6.0*((x + 1j*y)**3 - 0.25*x**3) \
				+ 1.0/24.0*(4.0*x*(1j*y)**3 + 6.0*(x**2)*(1j*y)**2 + 3.0*1j*y*(x**3) + 3.0/8.0*x**4) \
				+ 1.0/48.0*(4.0*(x**2)*(1j*y)**3 + 3.0*(x**3)*(1j*y)**2 + 0.75*(x**4)*1j*y) \
				+ 1.0/96.0*(2.0*(x**3)*(1j*y)**3 + 0.75*(x**4)*(1j*y)**2) \
				+ 1.0/384.0*(x**4)*(1j*y)**3

		s = np.abs(e)

		make_contourf(x, y, s, fig, ax, **self.properties)


class StabilityFunctionWrapper(DrawerWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader)

		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			coupling_method = data['coupling_method']
			xlim, nx = data['xlim'], data['nx']
			ylim, ny = data['ylim'], data['ny']
			drawer_properties = data['drawer_properties']

			self._core = StabilityFunction(
				coupling_method, xlim, nx, ylim, ny, properties=drawer_properties
			)
