#!/bin/bash
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

PYTHON_SCRIPT=make_plot_composite.py
JSON_FILE=plot_composite.json
SHOW=0
TIME_LEVELS_START=00
TIME_LEVELS_STEP=02
TIME_LEVELS_STOP=64
OUTPUT_ROOT=..\\/..\\/results\\/figures\\/isentropic_moist_rk3ws_si_fifth_order_upwind_pg2_nx41_ny41_nz60_dt45_nt640_gaussian_L50000_H1000_u15_rh90_turb_f_sed_evap\\/qr_xy_

for i in $(seq -w ${TIME_LEVELS_START} ${TIME_LEVELS_STEP} ${TIME_LEVELS_STOP})
do
	echo ""
	j=$(printf "%d" ${i})
	sed -i "s/.*tlevels.*/\t\"tlevels\": ${j},/g" config/${JSON_FILE}
	sed -i "s/.*save_dest.*/\t\"save_dest\": \"${OUTPUT_ROOT}${i}.eps\"/g" config/${JSON_FILE}
	python ${PYTHON_SCRIPT} config/${JSON_FILE} ${SHOW}
done
echo ""
