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

TASMANIA_ROOT=$(cd ..; pwd)
GT4PY_BRANCH=tasmania_migration
DOCKERFILE=dockerfile.tasmania
IMAGE_NAME=tasmania:master-ng
IMAGE_SAVE=tasmania_master_ng.tar

echo "About to clone the gridtools4py repository under '$PWD/gridtools4py', and check out the '$GT4PY_BRANCH' branch."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [ ! -d "gridtools4py" ]; then
		cd ..
		git submodule add https://github.com/eth-cscs/gridtools4py.git docker/gridtools4py
		cd docker
	fi

	cd gridtools4py
	git fetch
	git checkout $GT4PY_BRANCH
	git pull
	cd ..
fi

echo ""
echo "About to copy a minimal version of '$TASMANIA_ROOT' under '$PWD/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [ ! -d "tasmania" ]; then
		mkdir tasmania
	else
		rm -rf tasmania/*
	fi

	cp -r ../makefile tasmania
	cp -r ../notebooks tasmania
	cp -r ../README.md tasmania
	cp -r ../requirements.txt tasmania
	cp -r ../setup.cfg tasmania
	cp -r ../setup.py tasmania
	cp -r ../tasmania tasmania
	cp -r ../tests tasmania
fi

echo ""
echo "About to remove the tar archive '$PWD/$IMAGE_SAVE' (if existing)."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [ -f "$IMAGE_SAVE" ]; then
		rm -rf $IMAGE_SAVE
	fi
fi

echo ""
echo "About to build the image '$IMAGE_NAME' against the dockerfile '$PWD/dockerfiles/$DOCKERFILE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	cd .. && make distclean && cd docker 
	docker build --rm --build-arg uid=$(id -u) -f dockerfiles/$DOCKERFILE -t $IMAGE_NAME .
fi

echo ""
echo "About to delete the folder '$PWD/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	rm -rf tasmania
fi

echo ""
echo "About to save the image '$IMAGE_NAME' to the tar archive '$PWD/$IMAGE_SAVE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	docker save --output $IMAGE_SAVE $IMAGE_NAME
fi

