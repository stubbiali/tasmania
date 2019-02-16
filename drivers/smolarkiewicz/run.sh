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

NAMELISTS_FROM=(
	namelists/namelist_smolarkiewicz_ssus_1.py
	namelists/namelist_smolarkiewicz_ssus_2.py
	namelists/namelist_smolarkiewicz_sus_1.py
)

NAMELISTS_TO=(
	namelist_smolarkiewicz_ssus.py
	namelist_smolarkiewicz_ssus.py
	namelist_smolarkiewicz_sus.py
)

DRIVERS=(
	driver_namelist_smolarkiewicz_ssus.py
	driver_namelist_smolarkiewicz_ssus.py
	driver_namelist_smolarkiewicz_sus.py
)

for i in $(seq 0 $((${#NAMELISTS_FROM[@]} - 1))); do
	printf "Copy %s into %s.\n" ${NAMELISTS_FROM[i]} ${NAMELISTS_TO[i]}
	cp ${NAMELISTS_FROM[i]} ${NAMELISTS_TO[i]}
	printf "Run %s.\n\n" ${DRIVERS[i]}
	python ${DRIVERS[i]}
	printf "\n%s completed.\n\n" ${DRIVERS[i]}
done
