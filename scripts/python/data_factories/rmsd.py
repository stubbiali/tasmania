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
from data_factories.diagnostics_ledger import \
	OfflineDiagnosticComponentsLedger as ODCLedger
import tasmania as taz


#==================================================
# User inputs
#==================================================
filename1 = '../../data/isentropic_dry_rk3cosmo_fifth_order_upwind_rk2_' \
	'nx81_ny81_nz60_dt24_nt1800_gaussian_L50000_H1000_u15_f_sus.nc'
filename2 = '../../data/isentropic_dry_rk3cosmo_fifth_order_upwind_' \
	'nx81_ny81_nz60_dt24_nt1800_gaussian_L50000_H1000_u15_f_cc.nc'

field_name  = 'y_momentum_isentropic'
field_units = 'kg m^-1 K^-1 s^-1'

x1, x2 = slice(20, 61, None), slice(20, 61, None)
y1, y2 = slice(20, 61, None), slice(20, 61, None)
z1, z2 = slice(0, 60, None), slice(0, 60, None)


#==================================================
# Code
#==================================================
def get_grid():
	loader1 = LoaderFactory.factory(filename1)
	return loader1.get_grid()


def get_state(tlevel):
	loader1 = LoaderFactory.factory(filename1)
	loader2 = LoaderFactory.factory(filename2)

	grid1 = loader1.get_grid()
	grid2 = loader2.get_grid()

	if ODCLedger.get(__name__) is None:
		ODCLedger.register(
			__name__, 
			taz.RMSD(
				(grid1, grid2), {field_name: field_units},
				x=(x1, x2), y=(y1, y2), z=(z1, z2)
			)
		)
	rmsd = ODCLedger.get(__name__)

	tlevel = loader1.nt + tlevel if tlevel < 0 else tlevel

	state1, state2 = loader1.get_state(tlevel), loader2.get_state(tlevel)
	diagnostics = rmsd(state1, state2)
	state1.update(diagnostics)

	return state1


def get_initial_time():
	loader = LoaderFactory.factory(filename)
	return loader.get_state(0)['time']