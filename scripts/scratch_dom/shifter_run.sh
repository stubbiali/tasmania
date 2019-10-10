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

IMAGE_NAME=tasmania
IMAGE_TAG=develop
ROOT_DIR=tasmania-develop

module load daint-mc
module load shifter-ng

shifter run \
	--mount=type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
	--mount=type=bind,src=$HOME,dst=/home/tasmania-user/mount-point \
	--writable-volatile=/home/tasmania-user \
	load/library/$IMAGE_NAME:$IMAGE_TAG \
	bash -c \
		"cp -r $ROOT_DIR/tasmania/python ~/tasmania/tasmania; \
		 cp -r $ROOT_DIR/tasmania/__init__.py ~/tasmania/tasmania; \
		 cp -r $ROOT_DIR/tasmania/conf.py ~/tasmania/tasmania; \
		 export XAUTHORITY=/home/tasmania-user/mount-point/.Xauthority; \
		 bash"
