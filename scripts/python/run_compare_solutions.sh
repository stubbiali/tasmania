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
python compare_solutions.py \
  -f1 ../../data/burgers_validation/burgers_ssus_debug.nc \
  -f2 ../../data/burgers_validation/burgers_ssus_numpy.nc \
  -t1 0 401 1 \
  -t2 0 401 1

python compare_solutions.py \
  -f1 ../../data/burgers_validation/burgers_ssus_debug.nc \
  -f2 ../../data/burgers_validation/burgers_ssus_gtx86.nc \
  -t1 0 401 1 \
  -t2 0 401 1

python compare_solutions.py \
  -f1 ../../data/burgers_validation/burgers_ssus_debug.nc \
  -f2 ../../data/burgers_validation/burgers_ssus_gtmc.nc \
  -t1 0 401 1 \
  -t2 0 401 1

python compare_solutions.py \
  -f1 ../../data/burgers_validation/burgers_ssus_debug.nc \
  -f2 ../../data/burgers_validation/burgers_ssus_gtcuda.nc \
  -t1 0 401 1 \
  -t2 0 401 1
