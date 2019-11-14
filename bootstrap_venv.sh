#!/bin/bash

MODULES=( )
# MODULES=( cray-python/3.6.5.7 cudatoolkit )
PYTHON=python3.7   # use 'python3.7' when possible
CUDA=
VENV=venv
FRESH_INSTALL=1

function install()
{
  source $VENV/bin/activate && \
    pip install wheel && \
    pip install -e . && \
    pip install -e docker/external/gt4py[$CUDA] || \
      pip install -e docker/external/gt4py && \
    python docker/external/gt4py/setup.py install_gt_sources && \
    pip install -e docker/external/sympl

  # On OSX only:
  # change matplotlib backend from macosx to TkAgg
  # cat $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc | \
  #   sed -e 's/^backend.*: macosx/backend : TkAgg/g' > /tmp/.matplotlibrc && \
  #   cp /tmp/.matplotlibrc $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc && \
  #   rm /tmp/.matplotlibrc
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
