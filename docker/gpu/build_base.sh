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
DOCKERFILE=$CALL_DIR/gpu/dockerfiles/dockerfile.base
IMAGE_NAME=tasmania:gpu-base
IMAGE_SAVE=$CALL_DIR/images/tasmania_gpu_base.tar

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
	docker build --rm -f $DOCKERFILE -t $IMAGE_NAME .
fi

echo ""
echo "About to save the image '$IMAGE_NAME' to the tar archive '$IMAGE_SAVE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	docker save --output $IMAGE_SAVE $IMAGE_NAME
fi