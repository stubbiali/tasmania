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
import matplotlib.pyplot as plt
import numpy as np
import tasmania.python.plot.plot_utils as pu 

#==================================================
# User inputs
#==================================================
field_name = 'v'
eps = 0.1
t = 1.0

xlim = [0, 1]
nx = 101
ylim = [0, 1]
ny = 101

figure_properties = {
	'fontsize': 16,
	'figsize': (8, 6),
	'tight_layout': True,
	'tight_layout_rect': None, #(0.0, 0.0, 0.7, 1.0),
}

axes_properties = {
	'fontsize': 16,
	'title_center': '',
	'title_left': '',
	'title_right': '',
	'x_label': '$x$', #'Time (UTC)',
	'x_lim': (xlim[0], xlim[1]), #(-190, 210),
	'invert_xaxis': False,
	'x_scale': None,
	'x_ticks': [0.0, 0.25, 0.5, 0.75, 1.0],
	'x_ticklabels': None,
	'xaxis_minor_ticks_visible': False,
	'xaxis_visible': True,
	'y_label': '$y$',
	'y_lim': (ylim[0], ylim[1]),
	'invert_yaxis': False,
	'y_scale': None,
	'y_ticks': [0.0, 0.25, 0.5, 0.75, 1.0],
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

contourf_properties = {
	'fontsize': 16,
	'cmap_name': 'jet',
	'cbar_on': True,
	'cbar_levels': 17,
	'cbar_ticks_step': 2,
	'cbar_ticks_pos': 'interface',
	'cbar_center': 0.0,
	'cbar_half_width': 0.0025, 
	'cbar_x_label': '',
	'cbar_y_label': '$v$-velocity',
	'cbar_title': '',
	'cbar_orientation': 'vertical',
	'cbar_ax': None,
 	'cbar_format': '%3.2e',
	'draw_vertical_levels': False,
}

#==================================================
# Code
#==================================================
if __name__ == '__main__':
	xv = np.linspace(xlim[0], xlim[1], nx)
	yv = np.linspace(ylim[0], ylim[1], ny)
	x, y = np.meshgrid(xv, yv)

	if field_name == 'u':
		field = - 2.0 * eps * 2.0 * np.pi * np.exp(- 5 * np.pi**2 * eps * t) * \
			np.cos(2.0 * np.pi * x) * np.sin(np.pi * y) / \
			(2.0 + np.exp(- 5.0 * np.pi**2 * eps * t) * np.sin(2.0 * np.pi * x) * 
    		 np.cos(np.pi * y))
	else:
		field = - 2.0 * eps * np.pi * np.exp(- 5 * np.pi**2 * eps * t) * \
			np.sin(2.0 * np.pi * x) * np.cos(np.pi * y) / \
			(2.0 + np.exp(- 5.0 * np.pi**2 * eps * t) * np.sin(2.0 * np.pi * x) * 
    		 np.cos(np.pi * y))

	fig, ax = pu.get_figure_and_axes(**figure_properties)
	pu.make_contourf(xv, yv, field, fig, ax, **contourf_properties)
	pu.set_axes_properties(ax, **axes_properties)
	pu.set_figure_properties(fig, **figure_properties)
	plt.show()