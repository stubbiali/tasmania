#!/bin/bash

IMAGE_NAME=tasmania:master

echo "About to build the container image '$MAGE_NAME' for tasmania." 
read -n 1 -s -r -p "Press any key to continue, or Ctrl-C to exit."

cp ../requirements.txt .

if [ ! -d "gridtools4py" ]; then
	git clone https://github.com/eth-cscs/gridtools4py.git
fi

cd gridtools4py
git checkout merge_ubbiali
cd ..

docker build --rm --build-arg uid=$(id -u) -t tasmania-master .

rm requirements.txt
