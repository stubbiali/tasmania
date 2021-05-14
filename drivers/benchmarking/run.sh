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

DRIVER_DIR=isentropic_moist
DRIVER_NAME=driver_namelist_fc.py
BACKENDS=(
#  numpy
#  cupy
  gt4py:gtx86
  gt4py:gtmc
#  gt4py:gtc:gt:cpu_ifirst
#  gt4py:gtc:gt:cpu_kfirst
  gt4py:gtcuda
#  gt4py:gtc:cuda
#  gt4py:gtc:gt:gpu
)
NRUNS=5

CDIR=$PWD

function singlerun_nolog()
{
  echo $1
  cd $DRIVER_DIR && python $DRIVER_NAME -b $BACKEND --no-log && cd $CDIR || return
  echo ""
}

function singlerun()
{
  echo $1 $2
  cd $DRIVER_DIR && python $DRIVER_NAME -b $BACKEND --no-log && cd $CDIR || return
  echo ""
}

for BACKEND in "${BACKENDS[@]}"; do
  singlerun_nolog $BACKEND || return
  for i in $(seq 1 $NRUNS); do
    singlerun $BACKEND $i || return
  done
done
