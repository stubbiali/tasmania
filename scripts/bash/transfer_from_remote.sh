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

# ==================================================
# User input
# ==================================================
REMOTE=daint
REMOTE_ROOT=/project/s299/subbiali/tasmania
LOCAL_ROOT=../..

# ==================================================
# Code
# ==================================================
FILES_TO_COPY=()
ALL_FILES=($(ssh $REMOTE ls $REMOTE_ROOT))

echo "About to transfer data from the remote folder $REMOTE:$REMOTE_ROOT to the local folder $LOCAL_ROOT."
echo "Only the files newer than the version at the destination will be transfered."
read -n 1 -s -r -p "Press ENTER to continue, or CTRL-C to exit."
echo ""

k=0
for i in $(seq 0 $((${#ALL_FILES[@]} - 1))); do
	echo ""
	read -n 1 -s -r -p "Do you wish to transfer $REMOTE:$REMOTE_ROOT/${ALL_FILES[i]}? [y/n]" key
	if [[ $key = "y" ]]; then
		FILES_TO_COPY[$k]=${ALL_FILES[i]}
		((k++))
	fi
done

echo ""
echo ""

for i in $(seq 0 $((${#FILES_TO_COPY[@]} - 1))); do
	#scp -r daint:$DAINT_ROOT/${FILES_TO_COPY[i]} $LOCAL_ROOT
	rsync -avuzhr -e ssh --progress $REMOTE:$REMOTE_ROOT/${FILES_TO_COPY[i]} $LOCAL_ROOT
done
