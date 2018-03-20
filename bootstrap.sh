#!/bin/bash

HOME_DIR=/home/vagrant
VENV_DIR=${HOME_DIR}/venv
TASMANIA_ROOT=${HOME_DIR}/tasmania
GRIDTOOLS_ROOT=${HOME_DIR}/tasmania/gridtools

#
# Install Boost-1.58.0
#
if [ ! -d "/usr/local/include/boost" ]; then
	wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.58.0%2F&ts=1446134333&use_mirror=kent'
	tar xvzf boost_1_58_0.tar.gz
	cd boost_1_58_0
	./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local
	./b2 install
	cd ..
	rm -rf boost_1_58_0 boost_1_58_0.tar.gz
fi

#
# Environment setup for regular user
#
if [ ! -v FIRST_PROVISIONING_DONE ]; then
	echo "                                                       						" >> ~/.bashrc
	echo "#                                                      						" >> ~/.bashrc
	echo "# Environment setup for Gridtools/Tasmania development 						" >> ~/.bashrc
	echo "#																				" >> ~/.bashrc
	echo "export FIRST_PROVISIONING_DONE=true                                			" >> ~/.bashrc
	echo "export CXX=/usr/bin/g++                                						" >> ~/.bashrc
	echo "export BOOST_ROOT=/usr/local                           						" >> ~/.bashrc
	echo "export CUDATOOLKIT_HOME=/usr/local/cuda-7.0            						" >> ~/.bashrc
	echo "export TASMANIA_ROOT=$PWD/tasmania            		 						" >> ~/.bashrc
	echo "export GRIDTOOLS_ROOT=$PWD/tasmania/gridtools 								" >> ~/.bashrc
	echo "export LD_LIBRARY_PATH=/usr/local/lib                  						" >> ~/.bashrc
	echo "export PATH=${PATH}:${CUDATOOLKIT_HOME}/bin            						" >> ~/.bashrc
	echo "export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/tasmania:$PWD/tasmania/gridtools		" >> ~/.bashrc
fi
