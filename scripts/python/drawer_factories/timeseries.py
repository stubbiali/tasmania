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
from loader import LoaderFactory
import tasmania as taz


#==================================================
# User inputs
#==================================================
field_name  = 'rmsd_of_y_momentum_isentropic'
field_units = 'kg m^-1 K^-1 s^-1'

time_mode     = 'elapsed'
init_time     = datetime(year=1992, month=2, day=20, hour=0)
time_units    = 'hr'
time_on_xaxis = True

drawer_properties = {
	'fontsize': 16,
	'linestyle': '-',
	'linewidth': 1.5,
	'linecolor': 'blue',
	'marker': 'o',
	'markersize': 7,
	'markeredgewidth': 1,
	'markerfacecolor': 'blue',
	'markeredgecolor': 'blue',
	'legend_label': 'SUS'
}


#==================================================
# Code
#==================================================
def get_drawer(df_module):
	import_str = 'from {} import get_grid'.format(df_module)
	exec(import_str)
	grid = locals()['get_grid']()

	return taz.TimeSeries(
		grid, field_name, field_units,
		time_mode=time_mode, init_time=init_time,
		time_units=time_units, time_on_xaxis=time_on_xaxis,
		properties=drawer_properties
	)


def get_state(df_module, drawer, tlevel, axes_properties=None, print_time=None):
	import_str = 'from {} import get_state as df_get_state'.format(df_module)
	exec(import_str)

	tlevel = loader.nt + tlevel if tlevel < 0 else tlevel
	drawer_tlevel = len(drawer._data)-1

	# assumption: any drawer is always called on the same dataset
	if drawer_tlevel >= tlevel:
		for k in range(tlevel, drawer_tlevel+1):
			drawer._data.pop(k)
	else:
		for k in range(drawer_tlevel+1, tlevel):
			drawer(locals()['df_get_state'](k))

	return locals()['df_get_state'](tlevel)