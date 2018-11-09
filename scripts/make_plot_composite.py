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
from loader import LoaderComposite
import tasmania as taz


#
# User inputs
#
nrows = 1
ncols = 2

modules = [
	'make_contourf_xz',
	'make_contourf_xz_1',
]

fontsize = 16
figsize = (12, 7)
tight_layout = True


#
# Code
#
def get_artist(tlevel=None):
	slaves = []
	states = []

	for module in modules:
		import_str = 'from {} import get_artist as get_slave'.format(module)
		exec(import_str)
		slave, state = locals()['get_slave'](tlevel)
		slaves.append(slave)
		states.append(state)

	artist = taz.SubplotsAssembler(nrows, ncols, slaves, interactive=False,
								   fontsize=fontsize, figsize=figsize,
								   tight_layout=tight_layout)

	return artist, states


def get_loader():
	subloaders = []

	for module in modules:
		import_str = 'from {} import get_loader'.format(module)
		exec(import_str)
		subloaders.append(locals()['get_loader']())

	return LoaderComposite(subloaders)


if __name__ == '__main__':
	artist, states = get_artist()
	artist.store(states, show=True)
