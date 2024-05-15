#!/bin/bash
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
echo "namelist_ssus_0"
python driver_namelist_ssus.py -n namelists/namelist_ssus_0.py
echo ""

echo "namelist_ssus_1"
python driver_namelist_ssus.py -n namelists/namelist_ssus_1.py
echo ""

echo "namelist_ssus_2"
python driver_namelist_ssus.py -n namelists/namelist_ssus_2.py
echo ""

echo "namelist_ssus_3"
python driver_namelist_ssus.py -n namelists/namelist_ssus_3.py
echo ""
