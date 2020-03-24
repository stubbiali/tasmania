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
printf "\n namelist_ssus_4 \n"
python driver_namelist_ssus_ferk2.py -n namelists2d/namelist_ssus_ferk2_4.py
printf "\n namelist_ssus_5 \n"
python driver_namelist_ssus_ferk2.py -n namelists2d/namelist_ssus_ferk2_5.py