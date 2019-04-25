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
GT4PY_BRANCH=merge_ubbiali
IMAGE_NAME=tasmania:master
CONTAINER_NAME=$(openssl rand -hex 6)
IP=$(ifconfig en0 | grep 'inet ' | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | head -1)

echo "About to pull the branch '$GT4PY_BRANCH' of the gridtools4py repository."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [ ! -d "gridtools4py" ]; then
		git clone https://github.com/eth-cscs/gridtools4py.git
	fi

	cd gridtools4py
	git checkout $GT4PY_BRANCH
	cd ..
fi

echo ""
echo "About to run and connect to a containter named '$CONTAINER_NAME', spawn from the image '$IMAGE_NAME'." 
read -n 1 -s -r -p "Press CTRL-C to exit, or any other key to continue."
echo ""

ln -fs $DISPLAY /tmp/x11_display
open -a XQuartz
xhost + localhost
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:/tmp/x11_display &

docker run --rm									\
		   --privileged							\
		   -dit									\
		   -e DISPLAY=host.docker.internal:0 	\
		   -e XAUTHORITY=$XAUTHORITY 			\
		   -P									\
		   --name $CONTAINER_NAME				\
		   --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
		   --mount type=bind,src=$TASMANIA_ROOT/buffer,dst=/home/tasmania-user/tasmania/buffer \
		   --mount type=bind,src=$TASMANIA_ROOT/data,dst=/home/tasmania-user/tasmania/data \
		   --mount type=bind,src=$TASMANIA_ROOT/docker/gridtools4py,dst=/home/tasmania-user/tasmania/docker/gridtools4py \
		   --mount type=bind,src=$TASMANIA_ROOT/docs,dst=/home/tasmania-user/tasmania/docs \
		   --mount type=bind,src=$TASMANIA_ROOT/drivers,dst=/home/tasmania-user/tasmania/drivers \
		   --mount type=bind,src=$TASMANIA_ROOT/makefile,dst=/home/tasmania-user/tasmania/makefile \
		   --mount type=bind,src=$TASMANIA_ROOT/notebooks,dst=/home/tasmania-user/tasmania/notebooks \
		   --mount type=bind,src=$TASMANIA_ROOT/README.md,dst=/home/tasmania-user/tasmania/README.md \
		   --mount type=bind,src=$TASMANIA_ROOT/requirements.txt,dst=/home/tasmania-user/tasmania/requirements.txt \
		   --mount type=bind,src=$TASMANIA_ROOT/results,dst=/home/tasmania-user/tasmania/results \
		   --mount type=bind,src=$TASMANIA_ROOT/scripts,dst=/home/tasmania-user/tasmania/scripts \
		   --mount type=bind,src=$TASMANIA_ROOT/setup.cfg,dst=/home/tasmania-user/tasmania/setup.cfg \
		   --mount type=bind,src=$TASMANIA_ROOT/setup.py,dst=/home/tasmania-user/tasmania/setup.py \
		   --mount type=bind,src=$TASMANIA_ROOT/tasmania/__init__.py,dst=/home/tasmania-user/tasmania/tasmania/__init__.py \
		   --mount type=bind,src=$TASMANIA_ROOT/tasmania/conf.py,dst=/home/tasmania-user/tasmania/tasmania/conf.py \
		   --mount type=bind,src=$TASMANIA_ROOT/tasmania/python,dst=/home/tasmania-user/tasmania/tasmania/python \
		   --mount type=bind,src=$TASMANIA_ROOT/tests,dst=/home/tasmania-user/tasmania/tests \
		   $IMAGE_NAME					\
		   bash

docker exec -it $CONTAINER_NAME bash 
