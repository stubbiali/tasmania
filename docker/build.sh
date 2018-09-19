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

IMAGE_NAME=tasmania:master
GT4PY_BRANCH=merge_ubbiali

echo "About to build the container image $IMAGE_NAME for tasmania."
read -n 1 -r -p "Press any key to continue, or Ctrl-C to exit."

cp ../requirements.txt .

if [ ! -d "gridtools4py" ]; then
	git clone https://github.com/eth-cscs/gridtools4py.git
fi

cd gridtools4py
git checkout $GT4PY_BRANCH
cd ..

docker build --rm --build-arg uid=$(id -u) -t $IMAGE_NAME .

rm requirements.txt
