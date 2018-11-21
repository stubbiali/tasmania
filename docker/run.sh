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
CONTAINER_NAME=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 12 | head -n 1)

echo "About to fire up a containter named '$CONTAINER_NAME' from the image '$IMAGE_NAME'." 
read -n 1 -s -r -p "Press any key to continue, or Ctrl-C to exit."
echo ""

docker run --rm															\
		   --privileged													\
		   -dit															\
		   -e DISPLAY 													\
		   -e XAUTHORITY=$XAUTHORITY 									\
		   -e PYTHONPATH=/home/dockeruser/tasmania						\
		   -P															\
		   --name $CONTAINER_NAME										\
		   --mount type=bind,src=$PWD/..,dst=/home/dockeruser/tasmania	\
		   --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix		\
		   $IMAGE_NAME													\
		   bash

docker exec -it				\
			$CONTAINER_NAME \
			bash -c "set -ex; \
					 curl -LO https://bootstrap.pypa.io/get-pip.py; \
					 python get-pip.py --user; \
					 cd tasmania; \
					 make distclean; \
					 python -m pip install --user -e .; \
					 cd ..; \
					 bash"
