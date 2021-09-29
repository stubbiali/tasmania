#!/bin/bash
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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

MODULES=( )
# MODULES=( daint-gpu cray-python/3.6.5.7 cudatoolkit )
PYTHON=python3.9
PIP_UPGRADE=1
DISABLE_CEXT=1
GT4PY_EXTRAS=
VENV=venv
FRESH_INSTALL=0

function install()
{
  # activate environment
  source $VENV/bin/activate

  # upgrade pip and setuptools
  if [ "$PIP_UPGRADE" -gt 0 ]; then
    pip install --upgrade pip setuptools
  fi

  # install tasmania and required dependencies
  export DISABLE_TASMANIA_CEXT=$DISABLE_CEXT && \
	  pip install -e .

  # install xarray and sympl from source
  pip install -e docker/external/xarray
  pip install -e docker/external/sympl

  # install gt4py from source
  pip install -e docker/external/gt4py[$GT4PY_EXTRAS] || \
    pip install -e docker/external/gt4py

  # install gt sources
  python -m gt4py.gt_src_manager install

  # install development packages
  pip install -r requirements_dev.txt

  # deactivate environment
  deactivate

  # On OSX only: change matplotlib backend from macosx to TkAgg
  if [[ "$OSTYPE" == "darwin"* ]]; then
    cat $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc | \
      sed -e 's/^backend.*: macosx/backend : TkAgg/g' > /tmp/.matplotlibrc && \
      cp /tmp/.matplotlibrc $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc && \
      rm /tmp/.matplotlibrc
  fi
}

for MODULE in "${MODULES[@]}"
do
  module load $MODULE
done

if [ "$FRESH_INSTALL" -gt 0 ]
then
  echo -e "Creating new environment..."
  rm -rf $VENV
  $PYTHON -m venv $VENV
fi

install || deactivate

echo -e ""
echo -e "Command to activate environment:"
echo -e "\t\$ source $VENV/bin/activate"
echo -e ""
echo -e "Command to deactivate environment:"
echo -e "\t\$ deactivate"
echo -e ""
