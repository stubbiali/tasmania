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
TIME_LEVELS_START=25
TIME_LEVELS_STOP=90
OUTPUT_ROOT=..\\/..\\/results\\/isentropic_moist_rh90\\/isentropic_moist_rh90_

for i in $(seq ${TIME_LEVELS_START} ${TIME_LEVELS_STOP})
do
	echo ""
	sed -i "s/.*tlevels.*/\t\"tlevels\": ${i},/g" config/${JSON_FILE}
	sed -i "s/.*save_dest.*/\t\"save_dest\": \"${OUTPUT_ROOT}${i}.eps\"/g" config/${JSON_FILE}
	python ${PYTHON_SCRIPT} config/${JSON_FILE} ${SHOW}
done
echo ""
