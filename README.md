﻿<img align="right" src="taz.jpeg">

Tasmania
========

This is the repository for Tasmania, a Python library to ease the composition, configuration, and execution of Earth system models.

Motivation
----------

Available computer resources limit the resolution of climate projection and weather forecasting models, and thus pose a strict lower bound on the smallest scale of motion which can be directly resolved. Processes taking place at spatial scales smaller than the grid size, or occurring on timescales shorter than the model timestep, can not be explicitly represented. Notable examples include turbulent mixing in the planetary boundary layer, radiative transfer, convection, and cloud microphysics. As in many situations these processes exert a strong influence on the large-scale fields, they cannot simply be neglected. Rather, subgrid-scale phenomena are *parameterized*, i.e., their effect is expressed in terms of the resolved variables.

Although physical parameterizations have been largely studied in isolation, their combined effect, and their interaction with the underlying dynamical core (traditionally referred to as *physics-dynamics coupling*), has received significantly less attention. Among the pletora of motivations behind this deficiency, the lack of flexibility in most legacy codes, making them unsuitable to address these questions, have been playing a prominent role. 

Goal
----

Tasmania aims to be a high-level, modular and flexible framework to ease the composition, configuration, simulation and evaluation of finite difference numerical schemes for Earth system science. The library leverages [GridTools4Py](https://github.com/eth-cscs/gridtools4py), a complete set of Python bindings for the C++ template library [GridTools](https://github.com/eth-cscs/gridtools), developed at ETH/CSCS. GridTools furnishes a wide gamma of tools to implement stencil-based operations, thus findind a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., x86, MIC, and GPU. In addition to these, GridTools4Py supplies some Pythonic backends suitable for debugging and research purposes. On the other hand, GridTools's frontend, then GridTools4Py's interface, are hardware-agnostic, so that the user's code can be left unchanged when porting it to different architectures. This enables a complete separation of concerns between domain scientists - who can work in a familiar amd powerful development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

Installation and usage
----------------------

To clone this repository on your machine and place yourself on the current branch, from within a terminal run

	git clone https://github.com/eth-cscs/tasmania.git

We suggest to run any script or application leveraging Tasmania inside a [Docker](https://www.docker.com/) container, spawn from a provided image. To clone the GridTools4Py repository and create a local instance of the image (named `tasmania:master`), enter the folder `docker` and issue

	./build.sh

**Note**: both Tasmania and GridTools4Py repositories are *private*, so you should be given access to them in order to accomplish the steps above.

Later, launch

	./run.sh

to run a container in the background and get shell access to it. Before firing up an interactive `bash` session, the container installs Tasmania in editable mode via `pip`. 

You login the container's shell as `dockeruser` - a non-root user having the same `UID` of the host user who spined up the container. 

The local repository is bind mounted into the container; it can be write-accessed under `/home/dockeruser/tasmania`. This ensures that the latest version of Tasmania is always available inside the container. More generally, data in the repository can be seamlessy shared between the container and the host.   

`Ctrl+D` to exit the container's shell. To gain back shell access to the running container:

	docker exec -it CONTAINER_ID bash

`CONTAINER_ID` is the container identifier; it can be retrieved by inspecting the output of

	docker ps -a

Eventually, you can remove the container via

	docker stop CONTAINER_ID

**Disclaimer**: the container is given extended privileges to be able to run graphical applications, like generating a plot via [Matplotlib](https://matplotlib.org/) or running a [Jupyter](http://jupyter.org/) notebook. We are conscious of the fact that this is consider bad practice. Yet, it is the easiest way (to our knowledge) to allow containerized applications access the host desktop environment.
