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

NAMELISTS=(
	namelist_zhao_lcc.py
	namelist_zhao_lcc.py
	namelist_zhao_lcc.py
	namelist_zhao_lcc.py
)

FACTORS=(
	0
	1
	2
	3
)

DRIVERS=(
	driver_namelist_zhao_lcc.py
	driver_namelist_zhao_lcc.py
	driver_namelist_zhao_lcc.py
	driver_namelist_zhao_lcc.py
)

for i in $(seq 0 $((${#NAMELISTS[@]} - 1))); do
	cat ${NAMELISTS[i]} | sed -e "s/factor = .*/factor = ${FACTORS[i]}/g" > foo.py
	cp foo.py ${NAMELISTS[i]}
	python ${DRIVERS[i]}
	echo ""
done