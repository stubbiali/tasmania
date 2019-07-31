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
IMAGE_TAG=master
TASMANIA_ROOT=/project/s299/subbiali/tasmania

module load daint-mc
module load /apps/dom/SI/modulefiles/sarus/1.0.0-rc5-daint

mkdir -p buffer

sarus run \
	--tty \
	--mount=type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
	--mount=type=bind,src=$HOME,dst=/home/tasmania-user/mount-point/home \
	--mount=type=bind,src=$SCRATCH,dst=/home/tasmania-user/mount-point/scratch \
	--mount=type=bind,src=$PWD/buffer,dst=/home/tasmania-user/tasmania/buffer \
	--mount=type=bind,src=$TASMANIA_ROOT/data,dst=/home/tasmania-user/tasmania/data \
	--mount=type=bind,src=$TASMANIA_ROOT/docker/gridtools4py,dst=/home/tasmania-user/tasmania/docker/gridtools4py \
	--mount=type=bind,src=$TASMANIA_ROOT/docs,dst=/home/tasmania-user/tasmania/docs \
	--mount=type=bind,src=$TASMANIA_ROOT/drivers,dst=/home/tasmania-user/tasmania/drivers \
	--mount=type=bind,src=$TASMANIA_ROOT/makefile,dst=/home/tasmania-user/tasmania/makefile \
	--mount=type=bind,src=$TASMANIA_ROOT/notebooks,dst=/home/tasmania-user/tasmania/notebooks \
	--mount=type=bind,src=$TASMANIA_ROOT/README.md,dst=/home/tasmania-user/tasmania/README.md \
	--mount=type=bind,src=$TASMANIA_ROOT/requirements.txt,dst=/home/tasmania-user/tasmania/requirements.txt \
	--mount=type=bind,src=$TASMANIA_ROOT/results,dst=/home/tasmania-user/tasmania/results \
	--mount=type=bind,src=$TASMANIA_ROOT/scripts,dst=/home/tasmania-user/tasmania/scripts \
	--mount=type=bind,src=$TASMANIA_ROOT/setup.cfg,dst=/home/tasmania-user/tasmania/setup.cfg \
	--mount=type=bind,src=$TASMANIA_ROOT/setup.py,dst=/home/tasmania-user/tasmania/setup.py \
	--mount=type=bind,src=$TASMANIA_ROOT/tasmania/__init__.py,dst=/home/tasmania-user/tasmania/tasmania/__init__.py \
	--mount=type=bind,src=$TASMANIA_ROOT/tasmania/conf.py,dst=/home/tasmania-user/tasmania/tasmania/conf.py \
	--mount=type=bind,src=$TASMANIA_ROOT/tasmania/python,dst=/home/tasmania-user/tasmania/tasmania/python \
	--mount=type=bind,src=$TASMANIA_ROOT/tests,dst=/home/tasmania-user/tasmania/tests \
	load/library/$IMAGE_NAME:$IMAGE_TAG \
	bash -c \
		"export HOME=/home/tasmania-user/mount-point/home; \
		 export SCRATCH=/home/tasmania-user/mount-point/scratch; \
		 export XAUTHORITY=/home/tasmania-user/mount-point/home/.Xauthority; \
		 bash"
