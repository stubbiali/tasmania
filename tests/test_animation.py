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
import os
import pytest
import sys

from tasmania.python.plot.animation import Animation
from tasmania.python.plot.monitors import Plot
from tasmania.python.plot.profile import LineProfile


baseline_dir = 'baseline_images/py{}{}/test_animation'.format(
    sys.version_info.major, sys.version_info.minor
)
result_dir = 'result_images/py{}{}/test_animation'.format(
    sys.version_info.major, sys.version_info.minor
)


def test(isentropic_dry_data):
	# make sure the baseline directory does exist
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)

	# make sure the result directory does exist
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# make sure the baseline footage will exist at the end of this run
	filename = 'test_profile.mp4'
	baseline_video = os.path.join(baseline_dir, filename)
	result_video = os.path.join(result_dir, filename)
	save_dest = result_video if os.path.exists(baseline_video) else baseline_video

	# field to plot
	field_name  = 'x_velocity'
	field_units = 'km hr^-1'

	# grab data
	domain, grid_type, states = isentropic_dry_data
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid

	# indices identifying the cross-line to visualize
	y, z = int(grid.ny/2), -1

	# drawer properties
	drawer_properties = {
		'linecolor': 'blue',
		'linestyle': '-',
		'linewidth': 1.5,
	}

	# instantiate the drawer
	drawer = LineProfile(
		grid, field_name, field_units, y=y, z=z,
		axis_name='x', axis_units='km', properties=drawer_properties
	)

	# figure and axes properties
	figure_properties = {
		'fontsize': 16,
		'figsize': (7, 7),
		'tight_layout': True,
	}
	axes_properties = {
		'fontsize': 16,
		'title_left': '$x$-velocity [km hr$^{-1}$]',
		'x_label': '$x$ [km]',
		#'x_lim': [0, 500],
		'y_lim': [0, 100],
		'grid_on': True,
	}

	# instantiate the monitor
	monitor = Plot(
		drawer, interactive=False, figure_properties=figure_properties,
		axes_properties=axes_properties
	)

	# create the animation
	animation = Animation(monitor, print_time='elapsed', fps=15)
	for state in states:
		animation.store(state)
	for state in states:
		animation.store(state)
	for state in states:
		animation.store(state)
	for state in states:
		animation.store(state)
	animation.run(save_dest=save_dest)

	# asserts
	assert os.path.exists(save_dest)


if __name__ == '__main__':
	pytest.main([__file__])
