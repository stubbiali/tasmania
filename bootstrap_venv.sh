#!/bin/bash

PYTHON=python3.7

function install()
{
  source venv/bin/activate && \
	  pip install -e . && \
	  pip install -e docker/external/gridtools4py && \
	  python docker/external/gridtools4py/setup.py install_gt_sources && \
	  deactivate
}

if [[ ! -d "venv" ]]; then
	virtualenv --python=$PYTHON venv
fi

install || deactivate
