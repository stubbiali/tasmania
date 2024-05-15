#!/bin/bash
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
TASMANIA_ROOT="$(dirname "$TASMANIA_ROOT")"
DOCKER_DIR=$TASMANIA_ROOT/docker
EXTERNAL_DIR=$DOCKER_DIR/external
GT4PY_BRANCH=master
SYMPL_BRANCH=ubbiali
DOCKERFILE=$CDIR/dockerfiles/dockerfile.tasmania
IMAGE_NAME=tasmania:cpu
IMAGE_SAVE=$DOCKER_DIR/images/tasmania_cpu.tar

echo "About to clone the GridTools/gt4py repository under"
echo "'$EXTERNAL_DIR/gt4py',"
echo "and check out the '$GT4PY_BRANCH' branch."
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
	cd "$DOCKER_DIR" || return
fi

echo ""
echo "About to clone the stubbiali/sympl repository under"
echo "'$EXTERNAL_DIR/sympl',"
echo "and check out the '$SYMPL_BRANCH' branch."
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
	cd "$DOCKER_DIR" || return
fi

echo ""
echo "About to copy a minimal version of"
echo "'$TASMANIA_ROOT'"
echo "under '$TASMANIA_ROOT/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
  echo ""

	if [[ ! -d "tasmania" ]]; then
		mkdir $DOCKER_DIR/tasmania
	else
		rm -rf $DOCKER_DIR/tasmania/*
	fi

	cp -r $TASMANIA_ROOT/makefile $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/notebooks $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/README.md $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/requirements.txt $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/setup.cfg $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/setup.py $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/tasmania $DOCKER_DIR/tasmania
	cp -r $TASMANIA_ROOT/tests $DOCKER_DIR/tasmania
fi

echo ""
echo "About to remove the tar archive"
echo "'$IMAGE_SAVE' (if existing)."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	if [[ -f "$IMAGE_SAVE" ]]; then
		rm -rf $IMAGE_SAVE
	fi
fi

echo ""
echo "About to build the image '$IMAGE_NAME' against the dockerfile"
echo "'$DOCKERFILE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
  echo ""
	cd "$TASMANIA_ROOT" || return
	make distclean
	cd "$DOCKER_DIR" || return
  docker build --rm \
    --build-arg uid="$(id -u)" \
    --build-arg gt4py_branch="$GT4PY_BRANCH" \
    --build-arg sympl_branch="$SYMPL_BRANCH" \
    -f "$DOCKERFILE" -t $IMAGE_NAME .
fi

echo ""
echo "About to delete the folder '$DOCKER_DIR/tasmania'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	rm -rf "$DOCKER_DIR/tasmania"
fi

echo ""
echo "About to save the image '$IMAGE_NAME' to the tar archive"
echo "'$IMAGE_SAVE'."
read -n 1 -s -r -p "Press ENTER to continue, CTRL-C to exit, or any other key to bypass this step." key
echo ""

if [[ $key = "" ]]; then
	docker save --output $IMAGE_SAVE $IMAGE_NAME
fi
