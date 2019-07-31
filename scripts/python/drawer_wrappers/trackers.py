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


class TimeSeriesWrapper(DrawerWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader)

		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			field_name = data['field_name']
			field_units = data['field_units']
			x = data['x']
			y = data['y']
			z = data['z']
			time_mode = data['time_mode']
			init_time = datetime(
				year=data['init_time']['year'],
				month=data['init_time']['month'],
				day=data['init_time']['day'],
				hour=data['init_time'].get('hour', 0),
				minute=data['init_time'].get('minute', 0),
				second=data['init_time'].get('second', 0)
			)
			time_units = data['time_units']
			time_on_xaxis = data['time_on_xaxis']
			drawer_properties = data['drawer_properties']

			self._core = taz.TimeSeries(
				loader.get_grid(), field_name, field_units, x, y, z,
				time_mode=time_mode, init_time=init_time,
				time_units=time_units, time_on_xaxis=time_on_xaxis,
				properties=drawer_properties
			)

	def get_state(self, tlevel):
		tlevel = self._loader.get_nt() + tlevel if tlevel < 0 else tlevel
		drawer_tlevel = len(self._core._data)-1

		if drawer_tlevel >= tlevel:
			for k in range(tlevel, drawer_tlevel+1):
				self._core._data.pop(k)
		else:
			for k in range(drawer_tlevel+1, tlevel):
				self._core(self._loader.get_state(k))

		return self._loader.get_state(tlevel)


class LazyTimeSeriesWrapper(TimeSeriesWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader, json_filename)

	def get_state(self, tlevel):
		tlevel = self._loader.get_nt() + tlevel if tlevel < 0 else tlevel
		return self._loader.get_state(tlevel)


class HovmollerDiagramWrapper(DrawerWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader)

		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			field_name = data['field_name']
			field_units = data['field_units']

			x = data['x']
			y = data['y']
			z = data['z']

			axis_name = data['axis_name']
			axis_units = data['axis_units']
			axis_x = data['axis_x']
			axis_y = data['axis_y']
			axis_z = data['axis_z']

			time_mode = data['time_mode']
			init_time = datetime(
				year=data['init_time']['year'],
				month=data['init_time']['month'],
				day=data['init_time']['day'],
				hour=data['init_time'].get('hour', 0),
				minute=data['init_time'].get('minute', 0),
				second=data['init_time'].get('second', 0)
			)
			time_units = data['time_units']

			drawer_properties = data['drawer_properties']

			self._core = taz.HovmollerDiagram(
				loader.get_grid(), field_name, field_units, x=x, y=y, z=z,
				axis_name=axis_name, axis_units=axis_units,
				axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
				time_mode=time_mode, init_time=init_time, time_units=time_units,
				properties=drawer_properties
			)

	def get_state(self, tlevel):
		tlevel = self._loader.get_nt() + tlevel if tlevel < 0 else tlevel
		drawer_tlevel = len(self._core._time)-1

		if drawer_tlevel >= tlevel > 0:
			for k in range(tlevel, drawer_tlevel+1):
				self._core._time.pop(k)
			if self._core._txaxis:
				self._core._data = self._core._data[:tlevel+1, :]
			else:
				self._core._data = self._core._data[:, :tlevel+1]
		else:
			for k in range(drawer_tlevel, tlevel):
				self._core(self._loader.get_state(k))

		return self._loader.get_state(tlevel)


class LazyHovmollerDiagramWrapper(HovmollerDiagramWrapper):
	def __init__(self, loader, json_filename):
		super().__init__(loader, json_filename)

	def get_state(self, tlevel):
		tlevel = self._loader.get_nt() + tlevel if tlevel < 0 else tlevel
		return self._loader.get_state(tlevel)
