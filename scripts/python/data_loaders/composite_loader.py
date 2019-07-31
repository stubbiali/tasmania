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
import json


class CompositeLoader:
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			self._slaves = []
			for slave_data in data['loaders']:
				slave_module = slave_data['module']
				slave_classname = slave_data['classname']
				slave_config = slave_data['config']

				import_str = "from {} import {}".format(slave_module, slave_classname)
				exec(import_str)
				self._slaves.append(locals()[slave_classname](slave_config))

	def get_grid(self):
		return tuple(slave.get_grid() for slave in self._slaves)

	def get_nt(self):
		return tuple(slave.get_nt() for slave in self._slaves)

	def get_initial_time(self):
		return tuple(slave.get_initial_time() for slave in self._slaves)

	def get_state(self, tlevels):
		tlevels = (tlevels, )*len(self._slaves) if isinstance(tlevels, int) else tlevels
		return tuple(slave.get_state(tlevel) for slave, tlevel in zip(self._slaves, tlevels))