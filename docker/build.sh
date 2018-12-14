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

GT4PY_BRANCH=merge_ubbiali
IMAGE_NAME=tasmania:master
IMAGE_SAVE=tasmania_master.tar

echo "About to clone the gridtools4py repository under $PWD/gridtools4py, and check out the '$GT4PY_BRANCH' branch."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	cp ../requirements.txt .

	if [ ! -d "gridtools4py" ]; then
		git clone https://github.com/eth-cscs/gridtools4py.git
	fi

	cd gridtools4py
	git checkout $GT4PY_BRANCH
	cd ..
fi

echo ""
echo "About to remove the tar archive $PWD/$IMAGE_SAVE (if existing)."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	rm $PWD/$IMAGE_SAVE
fi

echo ""
echo "About to build the Docker image '$IMAGE_NAME' for tasmania."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	cd .. && make distclean && cd docker 
	docker build --rm --build-arg uid=$(id -u) -t $IMAGE_NAME .
fi

echo ""
echo "About to save the Docker image '$IMAGE_NAME' to the tar archive $PWD/$IMAGE_SAVE."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	docker save --output $IMAGE_SAVE $IMAGE_NAME
fi
