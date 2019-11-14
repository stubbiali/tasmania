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
rm -rf  burgers/conf.py     burgers/utils.py    \
        dwarfs/conf.py      dwarfs/utils.py     \
        framework/conf.py   framework/utils.py  \
        grids/conf.py       grids/utils.py      \
        isentropic/conf.py  isentropic/utils.py \
        physics/conf.py     physics/utils.py    \
        plot/conf.py        plot/utils.py       \
        utils/conf.py       utils/utils.py

ln -s   $PWD/conf.py    $PWD/burgers/conf.py
ln -s   $PWD/conf.py    $PWD/dwarfs/conf.py
ln -s   $PWD/conf.py    $PWD/framework/conf.py
ln -s   $PWD/conf.py    $PWD/grids/conf.py
ln -s   $PWD/conf.py    $PWD/isentropic/conf.py
ln -s   $PWD/conf.py    $PWD/physics/conf.py
ln -s   $PWD/conf.py    $PWD/plot/conf.py
ln -s   $PWD/conf.py    $PWD/utils/conf.py

ln -s   $PWD/utils.py   $PWD/burgers/utils.py
ln -s   $PWD/utils.py   $PWD/dwarfs/utils.py
ln -s   $PWD/utils.py   $PWD/framework/utils.py
ln -s   $PWD/utils.py   $PWD/grids/utils.py
ln -s   $PWD/utils.py   $PWD/isentropic/utils.py
ln -s   $PWD/utils.py   $PWD/physics/utils.py
ln -s   $PWD/utils.py   $PWD/plot/utils.py
ln -s   $PWD/utils.py   $PWD/utils/utils.py
