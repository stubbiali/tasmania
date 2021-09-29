#!/bin/bash
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
echo "namelist_sus_0"
python driver_namelist_sus.py -n namelists/namelist_sus_0.py
echo ""

echo "namelist_sus_1"
python driver_namelist_sus.py -n namelists/namelist_sus_1.py
echo ""

echo "namelist_sus_2"
python driver_namelist_sus.py -n namelists/namelist_sus_2.py
echo ""

echo "namelist_sus_3"
python driver_namelist_sus.py -n namelists/namelist_sus_3.py
echo ""
