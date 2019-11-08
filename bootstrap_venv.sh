#!/bin/bash

# MODULES=( python_virtualenv/15.0.3 cray-python/3.6.5.7 cudatoolkit )
MODULES=( )
PYTHON=python3.7
CUDA=
VENV=venv
FRESH_INSTALL=1

function install()
{
  source $VENV/bin/activate && \
	  pip install -e . && \
	  pip install -e docker/external/gridtools4py[$CUDA] || \
	    pip install -e docker/external/gridtools4py && \
	  python docker/external/gridtools4py/setup.py install_gt_sources && \
	  pip install -e docker/external/sympl && \
	  deactivate

  # change matplotlib backend from macos to TkAgg
	cat $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc | \
	  sed -e 's/^backend.*: macos/backend : TkAgg/g' > /tmp/.matplotlibrc && \
	  cp /tmp/.matplotlibrc $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc && \
	  rm /tmp/.matplotlibrc
}

for MODULE in "${MODULES[@]}"
do
  module load $MODULE
done

if [ "$FRESH_INSTALL" -gt 0 ]
then
  rm -rf $VENV
  virtualenv --python=$PYTHON $VENV
fi

install || deactivate

