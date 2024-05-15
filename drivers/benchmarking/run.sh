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
DRIVER_DIR=burgers
DRIVER_NAMES=(
  driver_namelist_sus.py
  driver_namelist_ssus.py
)
BACKENDS=(
  numpy
)
NRUNS=3

CDIR=$PWD

function singlerun_nolog()
{
  echo "$1" "$2"
  cd $DRIVER_DIR && python "$1" -b "$2" --no-log && cd "$CDIR" || return
  echo ""
}

function singlerun()
{
  echo "$1" "$2" "$3"
  cd $DRIVER_DIR && python "$1" -b "$2" && cd "$CDIR" || return
  echo ""
}

for DRIVER_NAME in "${DRIVER_NAMES[@]}"; do
  for BACKEND in "${BACKENDS[@]}"; do
#    singlerun_nolog "$DRIVER_NAME" "$BACKEND" || return
    for i in $(seq 1 $NRUNS); do
      singlerun "$DRIVER_NAME" "$BACKEND" "$i" || return
    done
  done
done
