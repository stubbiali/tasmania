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

CALL_DIR=$HOME/Desktop/phd/tasmania-develop-gt4py-v0.5.0/docker
TASMANIA_ROOT=$HOME/Desktop/phd/tasmania-develop-gt4py-v0.5.0
EXTERNAL_DIR=$CALL_DIR/external
GT4PY_BRANCH=tasmania_migration
DOCKERFILE=$CALL_DIR/cpu/dockerfiles/dockerfile.tasmania
IMAGE_NAME=tasmania:cpu
IMAGE_SAVE=$CALL_DIR/images/tasmania_cpu.tar

echo "About to clone the gridtools4py repository under '$EXTERNAL_DIR/gridtools4py', and check out the '$GT4PY_BRANCH' branch."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [[ ! -d "$EXTERNAL_DIR/gridtools4py" ]]; then
		git submodule add https://github.com/eth-cscs/gridtools4py.git $EXTERNAL_DIR/gridtools4py
		git submodule update --init --recursive
	fi

	git submodule update --init --recursive
	cd $EXTERNAL_DIR/gridtools4py
	git checkout $GT4PY_BRANCH
	cd $CALL_DIR
fi

echo ""
echo "About to copy a minimal version of '$TASMANIA_ROOT' under '$CALL_DIR/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [[ ! -d "tasmania" ]]; then
		mkdir tasmania
	else
		rm -rf tasmania/*
	fi

	cp -r $TASMANIA_ROOT/makefile tasmania
	cp -r $TASMANIA_ROOT/notebooks tasmania
	cp -r $TASMANIA_ROOT/README.md tasmania
	cp -r $TASMANIA_ROOT/requirements.txt tasmania
	cp -r $TASMANIA_ROOT/setup.cfg tasmania
	cp -r $TASMANIA_ROOT/setup.py tasmania
	cp -r $TASMANIA_ROOT/tasmania tasmania
	cp -r $TASMANIA_ROOT/tests tasmania
fi

echo ""
echo "About to remove the tar archive '$IMAGE_SAVE' (if existing)."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [[ -f "$IMAGE_SAVE" ]]; then
		rm -rf $IMAGE_SAVE
	fi
fi

echo ""
echo "About to build the image '$IMAGE_NAME' against the dockerfile '$DOCKERFILE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	cd $TASMANIA_ROOT && make distclean && cd $CALL_DIR
	docker build --rm --build-arg uid=$(id -u) -f $DOCKERFILE -t $IMAGE_NAME .
fi

echo ""
echo "About to delete the folder '$CALL_DIR/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	rm -rf tasmania
fi

echo ""
echo "About to save the image '$IMAGE_NAME' to the tar archive '$IMAGE_SAVE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	docker save --output $IMAGE_SAVE $IMAGE_NAME
fi

