#!/bin/bash

IMAGE_NAME=tasmania:master
GT4PY_BRANCH=merge_ubbiali

echo "About to build the container image $IMAGE_NAME for tasmania."
read -n 1 -r -p "Press any key to continue, or Ctrl-C to exit."

cp ../requirements.txt .

if [ ! -d "gridtools4py" ]; then
	git clone https://github.com/eth-cscs/gridtools4py.git
fi

cd gridtools4py
git checkout $GT4PY_BRANCH
cd ..

docker build --rm --build-arg uid=$(id -u) -t $IMAGE_NAME .

rm requirements.txt
