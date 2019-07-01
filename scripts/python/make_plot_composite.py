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
from datetime import datetime
import json
import tasmania as taz
from tasmania.python.utils.utils import assert_sequence


class PlotCompositeWrapper:
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			self._plot_wrappers = []

			for plot_data in data['plots']:
				wrapper_module = plot_data['wrapper_module']
				wrapper_classname = plot_data['wrapper_classname']
				wrapper_config = plot_data['wrapper_config']

				import_str = 'from {} import {}'.format(wrapper_module, wrapper_classname)
				exec(import_str)
				self._plot_wrappers.append(locals()[wrapper_classname](wrapper_config))

			nrows = data['nrows']
			ncols = data['ncols']

			print_time = data['print_time']
			
			if print_time == 'elapsed' and 'init_time' in data:
				init_time = datetime(
					year=data['init_time']['year'],
					month=data['init_time']['month'],
					day=data['init_time']['day'],
					hour=data['init_time'].get('hour', 0),
					minute=data['init_time'].get('minute', 0),
					second=data['init_time'].get('second', 0)
				)
			else:
				init_time = None

			figure_properties = data['figure_properties']

			self._core = taz.PlotComposite(
				*(wrapper.get_artist() for wrapper in self._plot_wrappers),
				nrows=nrows, ncols=ncols, interactive=False,
				print_time=print_time, init_time=init_time,
				figure_properties=figure_properties
			)

			self._tlevels = data.get('tlevels', (0, )*len(self._plot_wrappers))
			self._save_dest = data.get('save_dest', None)

	def get_artist(self):
		return self._core

	def get_states(self, tlevels=None):
		tlevels = self._tlevels if tlevels is None else tlevels
		tlevels = (tlevels, )*len(self._plot_wrappers) if isinstance(tlevels, int) else tlevels
		assert_sequence(tlevels, reflen=len(self._plot_wrappers), reftype=int)

		states = tuple(plot_wrapper.get_states(tlevels) for plot_wrapper in self._plot_wrappers)

		return states

	def store(self, tlevels=None, fig=None, show=False):
		states = self.get_states(tlevels)
		return self._core.store(*states, fig=fig, save_dest=self._save_dest, show=show)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate a figure with multiple subplots.')
	parser.add_argument(
		'configfile', metavar='configfile', type=str, help='JSON configuration file.'
	)
	parser.add_argument(
		'show', metavar='show', type=int, help='1 to show the generated plot, 0 otherwise.'
	)
	args = parser.parse_args()
	plot_wrapper = PlotCompositeWrapper(args.configfile)
	plot_wrapper.store(show=args.show)

