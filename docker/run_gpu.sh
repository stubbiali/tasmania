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
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TASMANIA_ROOT="$(dirname "$CDIR")"
TASMANIA_ROOT="$(dirname "$CDIR")"
EXTERNAL_DIR=$TASMANIA_ROOT/docker/external
GT4PY_BRANCH=master
SYMPL_BRANCH=ubbiali
IMAGE_NAME=tasmania:gpu
CONTAINER_NAME=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 12 | head -n 1)

echo "About to pull the branch '$GT4PY_BRANCH' of the GridTools/gt4py repository."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
  echo ""

	if [[ ! -d "$EXTERNAL_DIR/gt4py" ]]; then
		git submodule add git@github.com:GridTools/gt4py.git "$EXTERNAL_DIR/gt4py"
		git submodule update --init --recursive
	fi

	git submodule update --init --recursive
	cd "$EXTERNAL_DIR/gt4py" || return
	git checkout $GT4PY_BRANCH
	git pull
	cd "$CDIR" || return
fi

echo ""
echo "About to pull the branch '$SYMPL_BRANCH' of the stubbiali/sympl repository."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
  echo ""

	if [[ ! -d "$EXTERNAL_DIR/sympl" ]]; then
		git submodule add git@github.com:stubbiali/sympl.git "$EXTERNAL_DIR/sympl"
		git submodule update --init --recursive
	fi

	git submodule update --init --recursive
	cd "$EXTERNAL_DIR/sympl" || return
	git checkout $SYMPL_BRANCH
	git pull
	cd "$CDIR" || return
fi

echo ""
echo "About to run and connect to a containter named '$CONTAINER_NAME',"
echo "spawn from the image '$IMAGE_NAME'."
read -n 1 -s -r -p "Press CTRL-C to exit, or any other key to continue."
echo ""

docker run --rm							\
		   -dit							\
		   -e DISPLAY 					\
		   -e XAUTHORITY=$XAUTHORITY 	\
		   -P							\
		   --device /dev/dri			\
		   --name $CONTAINER_NAME		\
		   --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
		   --mount type=bind,src=$TASMANIA_ROOT/buffer,dst=/home/tasmania-user/tasmania/buffer \
		   --mount type=bind,src=$TASMANIA_ROOT/data,dst=/home/tasmania-user/tasmania/data \
		   --mount type=bind,src=$TASMANIA_ROOT/docker/external/gt4py,dst=/home/tasmania-user/tasmania/docker/external/gt4py \
		   --mount type=bind,src=$TASMANIA_ROOT/docker/external/sympl,dst=/home/tasmania-user/tasmania/docker/external/sympl \
		   --mount type=bind,src=$TASMANIA_ROOT/docs,dst=/home/tasmania-user/tasmania/docs \
		   --mount type=bind,src=$TASMANIA_ROOT/drivers,dst=/home/tasmania-user/tasmania/drivers \
		   --mount type=bind,src=$TASMANIA_ROOT/makefile,dst=/home/tasmania-user/tasmania/makefile \
		   --mount type=bind,src=$TASMANIA_ROOT/notebooks,dst=/home/tasmania-user/tasmania/notebooks \
		   --mount type=bind,src=$TASMANIA_ROOT/README.md,dst=/home/tasmania-user/tasmania/README.md \
		   --mount type=bind,src=$TASMANIA_ROOT/requirements.txt,dst=/home/tasmania-user/tasmania/requirements.txt \
		   --mount type=bind,src=$TASMANIA_ROOT/scripts,dst=/home/tasmania-user/tasmania/scripts \
		   --mount type=bind,src=$TASMANIA_ROOT/setup.cfg,dst=/home/tasmania-user/tasmania/setup.cfg \
		   --mount type=bind,src=$TASMANIA_ROOT/setup.py,dst=/home/tasmania-user/tasmania/setup.py \
		   --mount type=bind,src=$TASMANIA_ROOT/tasmania/__init__.py,dst=/home/tasmania-user/tasmania/tasmania/__init__.py \
		   --mount type=bind,src=$TASMANIA_ROOT/tasmania/python,dst=/home/tasmania-user/tasmania/tasmania/python \
		   --mount type=bind,src=$TASMANIA_ROOT/tests,dst=/home/tasmania-user/tasmania/tests \
		   $IMAGE_NAME					\
		   bash

docker exec -it $CONTAINER_NAME bash 
