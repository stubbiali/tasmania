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

#NITER=1
#
#for i in $(seq 1 $NITER); do
#  echo "python driver_namelist_fc.py"
#  python driver_namelist_fc.py 2> /dev/null
#  echo ""
#
#  echo "python driver_namelist_lfc.py"
#  python driver_namelist_lfc.py 2> /dev/null
#  echo ""
#
#  echo "python driver_namelist_ps.py"
#  python driver_namelist_ps.py 2> /dev/null
#  echo ""
#
#  echo "python driver_namelist_sts.py"
#  python driver_namelist_sts.py 2> /dev/null
#  echo ""
#
#  echo "python driver_namelist_sus.py"
#  python driver_namelist_sus.py 2> /dev/null
#  echo ""
#
#  echo "python driver_namelist_ssus.py"
#  python driver_namelist_ssus.py 2> /dev/null
#  echo ""
#done

python driver_namelist_fc.py -n namelists/namelist_fc_0.py
echo ""
python driver_namelist_fc.py -n namelists/namelist_fc_1.py
echo ""
python driver_namelist_fc.py -n namelists/namelist_fc_2.py
echo ""
python driver_namelist_fc.py -n namelists/namelist_fc_3.py
echo ""

python driver_namelist_lfc.py -n namelists/namelist_lfc_0.py
echo ""
python driver_namelist_lfc.py -n namelists/namelist_lfc_1.py
echo ""
python driver_namelist_lfc.py -n namelists/namelist_lfc_2.py
echo ""

python driver_namelist_ps.py -n namelists/namelist_ps_0.py
echo ""
python driver_namelist_ps.py -n namelists/namelist_ps_1.py
echo ""
python driver_namelist_ps.py -n namelists/namelist_ps_2.py
echo ""

python driver_namelist_sts.py -n namelists/namelist_sts_0.py
echo ""
python driver_namelist_sts.py -n namelists/namelist_sts_1.py
echo ""
python driver_namelist_sts.py -n namelists/namelist_sts_2.py
echo ""

python driver_namelist_sus.py -n namelists/namelist_sus_0.py
echo ""
python driver_namelist_sus.py -n namelists/namelist_sus_1.py
echo ""
python driver_namelist_sus.py -n namelists/namelist_sus_2.py
echo ""

python driver_namelist_ssus.py -n namelists/namelist_ssus_0.py
echo ""
python driver_namelist_ssus.py -n namelists/namelist_ssus_1.py
echo ""
python driver_namelist_ssus.py -n namelists/namelist_ssus_2.py
