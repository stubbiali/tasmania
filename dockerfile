# Use Python3.5 runtime as parent image
FROM python:3.5-jessie

# Specify the author
MAINTAINER Stefano Ubbiali subbiali@phys.ethz.ch

# Set the working directory to /tasmania
WORKDIR /tasmania

# Add required files
ADD docker /tasmania/docker
ADD README.md /tasmania/README.md
ADD requirements.txt /tasmania/requirements.txt
ADD setup.py /tasmania/setup.py
ADD tasmania /tasmania/tasmania
ADD tests /tasmania/tests

# Install FFmpeg
RUN echo 'deb http://ftp.uk.debian.org/debian jessie-backports main' >> /etc/apt/sources.list && \
	apt-get update && \
	apt-get install -y ffmpeg

# Install chrome
RUN echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb stable main' >> /etc/apt/sources.list && \
	wget https://dl.google.com/linux/linux_signing_key.pub && \
	apt-key add linux_signing_key.pub && \
	apt-get update && \
	apt-get install -y google-chrome-stable 

# Install a miscellaneous of useful tools
RUN apt-get install -y cmake                \
					   libfreetype6-dev     \
					   libpng12-dev         \
					   git					\
					   graphviz				\
					   pkg-config           \
					   qt-sdk				\
					   vim					\
					   x11-apps				\
					--no-install-recommends

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
	apt-get install -y git-lfs && \
	git lfs install

# Install tasmania
RUN python -m pip install -e /tasmania

# Set Qt5 as matplotlib backend
RUN cat /usr/local/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc | \
	sed -e 's/^backend.*: TkAgg/backend : Qt5Agg/g' > /tmp/.matplotlibrc && \
	cp /tmp/.matplotlibrc /usr/local/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc

# Install GT4Py
RUN cd /tasmania/docker/gridtools4py && \
	git checkout merge_ubbiali && \
	python -m pip install --no-binary -e .

# Install sympl
RUN cd / && \
	git clone https://github.com/mcgibbon/sympl.git && \
	cd sympl && \
	git checkout master && \
	python -m pip install --no-binary -e .

# Install Boost-1.58.0
RUN wget -q -O boost_1_58_0.tar.gz 'http://downloads.sourceforge.net/project/boost/boost/1.58.0/boost_1_58_0.tar.gz?r=http%3A%2F%2Fsourceforge.net%2Fprojects%2Fboost%2Ffiles%2Fboost%2F1.58.0%2F&ts=1446134333&use_mirror=kent' && \
	tar xvzf boost_1_58_0.tar.gz && \
	cd boost_1_58_0 && \
	./bootstrap.sh --with-libraries=timer,system,chrono --exec-prefix=/usr/local && \
	./b2 install && \
	cd .. && \
	rm -rf boost_1_58_0 boost_1_58_0.tar.gz
ENV BOOST_ROOT /usr/local                           					

# Create a new (non-root) user called dockeruser, whose uid might be specified at building time
ARG uid=1000
RUN useradd -r -u $uid dockeruser

# Make home directory for dockeruser
RUN mkdir /home/dockeruser
RUN chmod a+w /home/dockeruser

# Switch to dockeruser
USER dockeruser

# Personalize bashrc configuration file
RUN cat /tasmania/docker/bashrc >> ~/.bashrc

# Set vimrc configuration file
RUN cp /tasmania/docker/vimrc ~/.vimrc

# Set useful environmental variables
ENV TERM xterm-256color
ENV QT_X11_NO_MITSHM 1
ENV CXX /usr/bin/g++                                					
ENV PYTHONPATH ${PYTHONPATH}:/tasmania
ENV PYTHONWARNINGS ignore:'numpy.dtype size changed':RuntimeWarning,ignore:'numpy.ufunc size changed':RuntimeWarning
#ENV TASMANIA_ROOT /tasmania            		 					
#ENV GRIDTOOLS_ROOT /tasmania/gridtools 							
#ENV LD_LIBRARY_PATH /usr/local/lib                  					
#ENV CUDATOOLKIT_HOME /usr/local/cuda-7.0            					
#ENV PATH $PATH:${CUDATOOLKIT_HOME}/bin            						
