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
from collections import OrderedDict
import tasmania as taz


#==================================================
# User inputs
<<<<<<< HEAD
#==================================================
modules = OrderedDict()
modules['drawer_factories.timeseries'] = 'data_factories.rmsd'
modules['drawer_factories.timeseries_1'] = 'data_factories.rmsd_1'

tlevel = 72
=======
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
>>>>>>> 7b4d10fc95a458b6ba4210d36b856a2d1aac6f3c

figure_properties = {
	'fontsize': 16,
	'figsize': (6, 6.5),
	'tight_layout': True,
	'tight_layout_rect': None, #(0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
	'fontsize': 16,
	# title
	'title_center': '',
	'title_left': '$x$-velocity [m s$^{-1}$]',
	'title_right': '08:00:00',
<<<<<<< HEAD
	# x-axis
	'x_label': '$x$-momentum [km]', #'Time (UTC)',
	'x_labelcolor': '',
=======
	'x_label': '$x$-momentum [km]', #'Time (UTC)',
>>>>>>> 7b4d10fc95a458b6ba4210d36b856a2d1aac6f3c
	'x_lim': None,  #(-200, 200), #(-190, 210),
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': None,  #range(-200, 201, 100), #(-190, -90, 10, 110, 210),
	'x_ticklabels': None,
	'x_tickcolor': '',
	'xaxis_minor_ticks_visible': False,
	'xaxis_visible': True,
	# y-axis
	'y_label': '$z$ [km]',
<<<<<<< HEAD
	'y_labelcolor': '',
=======
>>>>>>> 7b4d10fc95a458b6ba4210d36b856a2d1aac6f3c
	'y_lim': None,  #(-200, 200), #(-200, 200), # (0.01, 0.15),
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': None, #(-200, -100, 0, 100, 200),
	'y_ticklabels': None, #['{:1.1E}'.format(1e-4), '{:1.1E}'.format(1e-3), '{:1.1E}'.format(1e-2)],
	'y_tickcolor': '',
	'yaxis_minor_ticks_visible': False,
	'yaxis_visible': True,
	# z-axis
	'z_label': '',
	'z_labelcolor': '',
	'z_lim': None,
	'invert_zaxis': False,
	'z_scale': None,
	'z_ticks': None,
	'z_ticklabels': None,
	'z_tickcolor': '',
	'zaxis_minor_ticks_visible': True,
	'zaxis_visible': True,
	# legend
	'legend_on': False,
	'legend_loc': 'best', #'center left',
	'legend_bbox_to_anchor': None, #(1.04, 0.5),
	'legend_framealpha': 1.0,
	'legend_ncol': 1,
	# text
	'text': None, #'$w_{\\mathtt{FW}} = 0$',
	'text_loc': 'upper right',
	# grid
	'grid_on': True,
	'grid_properties': {'linestyle': ':'},
}

print_time = None  # 'elapsed', 'absolute'

save_dest = None


#==================================================
# Code
#==================================================
def get_plot():
	drawers = []

	for drawer_module in modules:
		import_str = 'from {} import get_drawer'.format(drawer_module)
		exec(import_str)
		drawer = locals()['get_drawer'](modules[drawer_module])
		drawers.append(drawer)

	return taz.Plot(drawers, False, figure_properties, axes_properties)


def get_states(tlevel, plot):
	drawers = plot.artists
	axes_properties = plot.axes_properties
	states  = []

	for drawer_module, drawer in zip(modules.keys(), drawers):
		import_str = 'from {} import get_state'.format(drawer_module)
		exec(import_str)
		state = locals()['get_state'](
			modules[drawer_module], drawer, tlevel, axes_properties, print_time
		)
		states.append(state)

	return states


if __name__ == '__main__':
	plot = get_plot()
	states = get_states(tlevel, plot)
	plot.store(states, save_dest=save_dest, show=True)
