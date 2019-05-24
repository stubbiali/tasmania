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
import tasmania as taz


class PlotWrapper:
	def __init__(self, json_filename):
		with open(json_filename, 'r') as json_file:
			data = json.load(json_file)

			self._drawer_wrappers = []

			for drawer_data in data['drawers']:
				loader_module = drawer_data['loader_module']
				loader_classname = drawer_data['loader_classname']
				loader_config = drawer_data['loader_config']

				import_str = "from {} import {}".format(loader_module, loader_classname)
				exec(import_str)
				loader = locals()[loader_classname](loader_config)

				wrapper_module = drawer_data['wrapper_module']
				wrapper_classname = drawer_data['wrapper_classname']
				wrapper_config = drawer_data['wrapper_config']

				import_str = "from {} import {}".format(wrapper_module, wrapper_classname)
				exec(import_str)
				self._drawer_wrappers.append(locals()[wrapper_classname](loader, wrapper_config))

			figure_properties = data['figure_properties']
			self._axes_properties = data['axes_properties']

			self._core = taz.Plot(
				*(wrapper.get_drawer() for wrapper in self._drawer_wrappers),
				interactive=False, figure_properties=figure_properties,
				axes_properties=self._axes_properties
			)

			self._tlevels = data.get('tlevels', 0)
			self._print_time = data.get('print_time', None)
			self._save_dest = data.get('save_dest', None)

	def get_artist(self):
		return self._core

	def get_states(self, tlevels=None):
		wrappers = self._drawer_wrappers

		tlevels = self._tlevels if tlevels is None else tlevels
		tlevels = (tlevels, )*len(wrappers) if isinstance(tlevels, int) else tlevels

		states = []

		for wrapper, tlevel in zip(wrappers, tlevels):
			states.append(wrapper.get_state(tlevel))

		if self._print_time == 'elapsed':
			for wrapper in wrappers:
				try:
					init_time = wrapper.get_initial_time()
					break
				except NotImplementedError:
					pass

			assert 'init_time' in locals()
			self._axes_properties['title_right'] = str(states[0]['time'] - init_time)
		elif self._print_time == 'absolute':
			self._axes_properties['title_right'] = str(states[0]['time'])

		return states

	def store(self, tlevels=None, fig=None, ax=None, show=False):
		states = self.get_states(tlevels)
		return self._core.store(*states, fig=fig, ax=ax, save_dest=self._save_dest, show=show)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate a figure with a single plot.')
	parser.add_argument(
		'configfile', metavar='configfile', type=str, help='JSON configuration file.'
	)
	args = parser.parse_args()
	plot_wrapper = PlotWrapper(args.configfile)
	plot_wrapper.store(show=True)
