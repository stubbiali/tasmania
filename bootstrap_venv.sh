#!/bin/bash

MODULES=( python_virtualenv/15.0.3 cray-python/3.6.5.7 cudatoolkit )
PYTHON=python3.6
CUDA=cuda101
VENV=venv

function install()
{
  source $VENV/bin/activate && \
	  pip install -e . && \
	  pip install -e docker/external/gridtools4py[$CUDA] && \
	  python docker/external/gridtools4py/setup.py install_gt_sources && \
	  pip install -e docker/external/sympl && \
	  deactivate
}

for MODULE in "${MODULES[@]}"
do
  module load $MODULE
done

rm -rf $VENV
virtualenv --python=$PYTHON $VENV
install || deactivate

