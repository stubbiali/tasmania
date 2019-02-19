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
from loader import LoaderFactory


#==================================================
# User inputs
#==================================================
filename = '../../data/isentropic_dry_rk3cosmo_fifth_order_upwind_' \
	'nx81_ny81_nz60_dt24_nt300_gaussian_L50000_H1000_u15_f_cc.nc'


#==================================================
# Code
#==================================================
def get_grid():
	loader = LoaderFactory.factory(filename)
	return loader.get_grid()


def get_state(tlevel):
	loader = LoaderFactory.factory(filename)
	return loader.get_state(tlevel)


def get_initial_time():
	loader = LoaderFactory.factory(filename)
	return loader.get_state(0)['time']