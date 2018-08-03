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
from matplotlib.testing.decorators import image_comparison
import os
import pytest

from conftest import isentropic_dry_data


@image_comparison(baseline_images=['test_plot_topography_3d'], extensions=['eps'], tol=0.15)
def test_plot_topography_3d():
	# Make sure the folder tests/baseline_images/test_plot_topography does exist
	baseline_dir = os.path.join(os.environ['TASMANIA_ROOT'],
								'tests/baseline_images/test_plot_topography')
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# Make sure the baseline image will exist at the end of this run
	baseline_img = os.path.join(baseline_dir, 'test_plot_topography_3d.eps')
	save_dest = None if os.path.exists(baseline_img) else baseline_img

	# Load and grab data
	grid, states = isentropic_dry_data()
	grid.update_topography(states[-1]['time'] - states[0]['time'])
	state = states[-1]

	# Plot properties
	plot_properties = {
		'fontsize': 16,
		'x_label': '$x$ [km]',
		'x_lim': [0, 500],
		'y_label': '$y$ [km]',
		'y_lim': [-250, 250],
		'z_label': '$z$ [km]',
		'z_lim': None,
	}

	# Plot function keyword arguments
	plot_function_kwargs = {
		'x_factor': 1e-3,
		'y_factor': 1e-3,
		'z_factor': 1e-3,
		'cmap_name': 'BrBG_r',
		'cbar_on': True,
		'cbar_orientation': 'vertical',
	}

	from tasmania.plot.topography import plot_topography_3d as plot_function
	from tasmania.plot.plot_monitors import Plot3d
	monitor = Plot3d(grid, plot_function, 'topography', interactive=False,
					 fontsize=16, figsize=[8, 7],
					 plot_properties=plot_properties, plot_function_kwargs=plot_function_kwargs)
	monitor.store(state, save_dest=save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
