#!/bin/bash

#
# Install GT4Py dependencies
#
apt-get update
apt-get install -y cmake                \
                   libfreetype6-dev     \
                   libpng12-dev         \
                   pkg-config           \
				   qt-sdk				\
				   vim					\
				   x11-apps				\
                --no-install-recommends

#
# Install Boost-1.58.0
#
wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.58.0%2F&ts=1446134333&use_mirror=kent'
tar xvzf boost_1_58_0.tar.gz
cd boost_1_58_0
./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local
./b2 install
cd ..
rm -rf boost_1_58_0 boost_1_58_0.tar.gz

#
# Install SIP-4.19.8
#
#wget -q -O sip-4.19.8.tar.gz 'https://sourceforge.net/projects/pyqt/files/sip/sip-4.19.8/sip-4.19.8.tar.gz/download'
#tar xvzf sip-4.19.8.tar.gz
#cd sip-4.19.8
#python configure.py
#make
#make install
#cd ..
#rm -rf sip-4.19.8 sip-4.19.8.tar.gz

#
# Download PyQt4-4.12.1
#
#wget -q -O PyQt4_gpl_x11-4.12.1.tar.gz 'https://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.12.1/PyQt4_gpl_x11-4.12.1.tar.gz/download'
#tar xvzf PyQt4_gpl_x11-4.12.1.tar.gz
#cd PyQt4_gpl_x11-4.12.1
#python configure-ng.py --confirm-license
#make
#make install
#cd ..
#rm -rf PyQt4_gpl_x11-4.12.1 PyQt4_gpl_x11-4.12.1.tar.gz

#
# Set the backend for matplotlib to Qt5
#
cat /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc | sed -e 's/^backend.*: TkAgg/backend : Qt5Agg/g' #| sed -e 's/^#backend.qt4.*/backend.qt4 : PyQt4/g' > /tmp/.matplotlibrc
cp /tmp/.matplotlibrc /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc

#
# Personalize bashrc configuration file
#
cat /tasmania/docker/bashrc >> ~/.bashrc

#
# Set vimrc configuration file
#
cp /tasmania/docker/vimrc ~/.vimrc
