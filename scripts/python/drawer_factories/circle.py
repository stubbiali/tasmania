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
drawer_properties = {
	'xy': (0.0, 0.0),
	'radius': 75,
	'linewidth': 2,
	'edgecolor': 'lime',
	'facecolor': 'none',
}


#==================================================
# Code
#==================================================
def get_drawer(df_module=None):
	return taz.Circle(drawer_properties)


def get_state(
	df_module=None, drawer=None, tlevel=None, axes_properties=None, print_time=None
):
	return {}