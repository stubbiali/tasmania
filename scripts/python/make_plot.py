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


#
# User inputs
#
modules = [
	#'make_contourf',
	#'make_rectangle',
	#'make_circle',
	#'make_topography2d',
	#'make_quiver',

	'make_profile',
	'make_profile_1',
	#'make_profile_2',

	#'make_contourf_analytical',

	#'make_contourf',
	#'make_topography1d',
]

tlevel = 1

figure_properties = {
	'fontsize': 16,
	'figsize': (6, 6.5),
	'tight_layout': True,
	'tight_layout_rect': None, #(0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '$x$-velocity [m s$^{-1}$]',
	'title_right': '08:00:00',
	'x_label': '$x$-momentum [km]', #'Time (UTC)',
	'x_lim': None,  #(-200, 200), #(-190, 210),
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': None,  #range(-200, 201, 100), #(-190, -90, 10, 110, 210),
	'x_ticklabels': None,
	'xaxis_minor_ticks_visible': False,
	'xaxis_visible': True,
	'y_label': '$z$ [km]',
	'y_lim': None,  #(-200, 200), #(-200, 200), # (0.01, 0.15),
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None, #(-200, -100, 0, 100, 200),
	'y_ticklabels': None, #['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
	'yaxis_minor_ticks_visible': False,
	'yaxis_visible': True,
	'z_label': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	'legend_on': False,
	'legend_loc': 'best', #'center left',
	'legend_bbox_to_anchor': None, #(1.04, 0.5),
	'legend_framealpha': 1.0,
	'legend_ncol': 1,
	'text': None, #'$w_{\\mathtt{FW}} = 0$',
	'text_loc': 'upper right',
	'grid_on': False,
	'grid_properties': {'linestyle': ':'},
}

print_time = None  # 'elapsed', 'absolute'

save_dest = None


#
# Code
#
def get_plot():
	drawers = []

	for module in modules:
		import_str = 'from {} import get_drawer'.format(module)
		exec(import_str)
		drawer = locals()['get_drawer']()
		drawers.append(drawer)

	return taz.Plot(drawers, False, figure_properties, axes_properties)


def get_states(tlevel, plot):
	drawers = plot.artists
	axes_properties = plot.axes_properties
	states  = []

	for module, drawer in zip(modules, drawers):
		import_str = 'from {} import get_state'.format(module)
		exec(import_str)
		state = locals()['get_state'](tlevel, drawer, axes_properties, print_time)
		states.append(state)

	return states


if __name__ == '__main__':
	plot = get_plot()
	states = get_states(tlevel, plot)
	plot.store(states, save_dest=save_dest, show=True)
