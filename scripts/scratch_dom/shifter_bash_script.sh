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
#SBATCH --job-name=ps0
#SBATCH --account=s299m
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=error/slurm-%j.err

# ==================================================
# User inputs
# ==================================================
VENV=py36  # options: py35, py36, py37
ROOT_DIR=tasmania-develop
DIR=drivers/burgers
SCRIPT_NAME=run_ps.sh  # options: any bash script under ROOT_DIR/DIR

# ==================================================
# Code
# ==================================================
module load daint-mc
module load shifter-ng

srun --unbuffered shifter run \
	--writable-volatile=/home/tasmania-user/$VENV \
	--writable-volatile=/home/tasmania-user/tasmania \
	load/library/tasmania:master \
	bash -c \
		"cp -r $ROOT_DIR/tasmania/python ~/tasmania/tasmania; \
		 cp -r $ROOT_DIR/tasmania/__init__.py ~/tasmania/tasmania; \
		 cp -r $ROOT_DIR/tasmania/conf.py ~/tasmania/tasmania; \
		 cd ~; \
		 . $VENV/bin/activate; \
		 cd $SCRATCH/$ROOT_DIR/$DIR; \
		 . $SCRIPT_NAME; \
		 deactivate; \
		 exit"
