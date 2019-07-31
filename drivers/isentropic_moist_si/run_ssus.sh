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

DRIVERS=(
	driver_namelist_ssus.py
	driver_namelist_ssus.py
	driver_namelist_ssus.py
	driver_namelist_ssus_sat.py
	driver_namelist_ssus_sat.py
)

NAMELISTS=(
	namelists/thetas300/namelist_ssus_2_0.py
	namelists/thetas300/namelist_ssus_2_2.py
	namelists/thetas300/namelist_ssus_2_5.py
	namelists/thetas300/namelist_ssus_2_2_sat.py
	namelists/thetas300/namelist_ssus_2_5_sat.py
)

for i in $(seq 0 $((${#DRIVERS[@]} - 1))); do
	printf "%s %s.\n" ${DRIVERS[i]} ${NAMELISTS[i]}
	python ${DRIVERS[i]} -n ${NAMELISTS[i]}
	printf "\n"
done
