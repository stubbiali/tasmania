#!/bin/bash

MODULES=( )
# MODULES=( daint-gpu cray-python/3.6.5.7 cudatoolkit )
PYTHON=python3.6
PIP_UPGRADE=1
DISABLE_CEXT=1
GT4PY_CUDA=
GT4PY_DAWN=1
VENV=venv
FRESH_INSTALL=1

function install()
{
  # activate environment
  source $VENV/bin/activate

  # upgrade pip
  if [ "$PIP_UPGRADE" -gt 0 ]; then
    pip install --upgrade pip
  fi

  # install tasmania and required dependencies
  export DISABLE_TASMANIA_CEXT=$DISABLE_CEXT && \
	  pip install -e .

  # install xarray and sympl from source
  pip install -e docker/external/xarray
  pip install -e docker/external/sympl

  # install gt4py from source
  if [ "$GT4PY_DAWN" -gt 0 ]; then
    pip install -e docker/external/gt4py[$GT4PY_CUDA,dawn] || \
      pip install -e docker/external/gt4py[dawn]
  else
    pip install -e docker/external/gt4py[$GT4PY_CUDA] || \
      pip install -e docker/external/gt4py
  fi

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
