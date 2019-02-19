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
import tasmania as taz


#==================================================
# User inputs
#==================================================
nrows = 1
ncols = 3

modules = [
	'make_plot',
	'make_plot_1',
	'make_plot_2',
]

tlevels = [16, 28, 40]

figure_properties = {
	'fontsize': 16,
	'figsize': (15, 7), # (15, 7), (10.5, 7)
	'tight_layout': True,
}

save_dest = None


#==================================================
# Code
#==================================================
def get_plot():
	subplots = []

	for module in modules:
		import_str = 'from {} import get_plot as get_subplot'.format(module)
		exec(import_str)
		subplots.append(locals()['get_subplot']())

	plot = taz.PlotComposite(
		nrows, ncols, subplots, interactive=False, figure_properties=figure_properties
	)

	return plot


def get_states(tlevels, plot):
	subplots = plot.artists
	states = []

	tlevels = (tlevels, ) * len(subplots) if isinstance(tlevels, int) else tlevels

	for i in range(len(subplots)):
		import_str = 'from {} import get_states'.format(modules[i])
		exec(import_str)
		states.append(locals()['get_states'](tlevels[i], subplots[i]))

	return states


if __name__ == '__main__':
	plot = get_plot()
	states = get_states(tlevels, plot)
	plot.store(states, save_dest=save_dest, show=True)
