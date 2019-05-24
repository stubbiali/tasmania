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
from .base_loader import BaseLoader
import json
from .mounter import DatasetMounter


class Loader(BaseLoader):
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)
			filename = ''.join(data['filename'])
			self._dsmounter = DatasetMounter(filename)

	def get_grid(self):
		return self._dsmounter.get_grid()

	def get_nt(self):
		return self._dsmounter.get_nt()

	def get_initial_time(self):
		return self._dsmounter.get_state(0)['time']

	def get_state(self, tlevel):
		return self._dsmounter.get_state(tlevel)


if __name__ == '__main__':
	json_filename = '/home/tasmania-user/tasmania/scripts/python/config/loader.json'

	_ = Loader(json_filename)
	_ = Loader(json_filename)
	obj = Loader(json_filename)

	nt = obj.get_nt()
	_ = obj.get_grid()
	_ = obj.get_state(nt-1)
	_ = obj.get_initial_time()

	print('All right!')