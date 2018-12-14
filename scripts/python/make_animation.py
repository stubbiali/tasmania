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
module = 'make_contourf_xz'

tlevels = range(0, 10, 2)

print_time = 'elapsed'  # 'elapsed', 'absolute'
fps = 10

save_dest = '../results/movies/smolarkiewicz/rk2_third_order_upwind_centered_' \
			'nx51_ny51_nz50_dt20_nt8640_flat_terrain.mp4'


#
# Code
#
if __name__ == '__main__':
	exec('from {} import get_plot as get_artist, get_states'.format(module))
	artist = locals()['get_artist']()

	engine = taz.Animation(artist, print_time=print_time, fps=fps)

	for t in tlevels:
		engine.store(locals()['get_states'](t, artist))

	engine.run(save_dest=save_dest)
