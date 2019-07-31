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
from datetime import datetime
import json
import tasmania as taz


class LineWrapper(DrawerWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader)

		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			field_name = data['field_name']
			field_units = data['field_units']
			x = data['x']
			y = data['y']
			z = data['z']
			xdata = data.get('xdata', None)
			ydata = data.get('ydata', None)
			drawer_properties = data['drawer_properties']

			self._core = taz.Line(
				loader.get_grid(), field_name, field_units, x, y, z,
				xdata=xdata, ydata=ydata, properties=drawer_properties
			)

	def get_state(self, tlevel):
		states = self._loader.get_state(tlevel)
		for state in states[:-1]:
			self._core(state)
		return states[-1]
