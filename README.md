<img align="right" src="taz.jpeg">

Tasmania
========

This is the repository for Tasmania, a Python library for Earth system science.

Motivation
----------

Available computer resources limit the resolution of climate projection and weather forecasting models, and thus pose a strict lower bound on the smallest scale of motion which can be directly resolved. Processes taking place at spatial scales smaller than the grid size, or occurring on timescales shorter than the model timestep, can not be explicitly represented. Notable examples include turbulent mixing in the planetary boundary layer, radiative transfer, convection, and cloud microphysics. As in many situations these processes exert a strong influence on the large-scale fields, they cannot simply be neglected. Rather, subgrid-scale phenomena are *parameterized*, i.e., their effect is expressed in terms of the resolved variables.

Although physical parameterizations have been largely studied in isolation, their combined effect, and their interaction with the underlying dynamical core (traditionally referred to as *physics-dynamics coupling*), has received significantly less attention. Among the pletora of motivations behind this deficiency, the lack of flexibility in most legacy codes, making them unsuitable to address these questions, have been playing a prominent role. 

Goal
----

Tasmania aims to be a high-level, modular and flexible framework to ease the composition, configuration, simulation and evaluation of finite difference numerical schemes for Earth system science. The library leverages [GridTools4Py](https://github.com/eth-cscs/gridtools4py), a complete set of Python bindings for the C++ template library [GridTools](https://github.com/eth-cscs/gridtools), developed at ETH/CSCS. GridTools furnishes a wide gamma of tools to implement stencil-based operations, thus findind a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., x86, MIC, and GPU. In addition to these, GridTools4Py supplies some Pythonic backends suitable for debugging and research purposes. On the other hand, GridTools's frontend, then GridTools4Py's interface, are hardware-agnostic, so that the user's code can be left unchanged when porting it to different architectures. This enables a complete separation of concerns between domain scientists - who can work in a familiar amd powerful development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

Installation
------------

We suggest to run any application relying on Tasmania inside a [Docker](https://www.docker.com/) container spawn from a provided image. To create a local instance of the image, from the repository root directory issue:

	docker build --build-arg uid=$(id -u) -t tasmania-ci docker

Later, type

	docker run --rm -dit -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/tasmania tasmania-ci

to run the container in detached mode. The flags `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` are required only if one wants to generate plots via [Matplotlib](https://matplotlib.org/). If this is not the case, they can be omitted. The flag `-v $PWD:/tasmania` attaches the current directory to the container *volume*, so that the folder `/tasmania` within the container will be in sync with the repository root directory.

When successful, the previous command returns a long sequence of letters and digits, representing the container identifier. This is required to gain shell access to the running container:

	docker exec -it CONTAINER_ID bash

Here, `CONTAINER_ID` is the container identifier.
