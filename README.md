<img align="right" src="taz.jpeg">

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

Installation
------------

To clone this repository on your machine and place yourself on the current branch, from within a terminal run:

	git clone https://github.com/eth-cscs/tasmania.git

Then, enter the folder `docker/` and type

	git clone https://github.com/eth-cscs/gridtools4py.git
	cd gridtools4py
	git checkout merge_ubbiali

to clone the GridTools4Py repository. **Note**: both Tasmania and GridTools4Py repositories are *private*, so you should be given access to them in order to accomplish the steps above.

We suggest to run any script or application relying on Tasmania inside a [Docker](https://www.docker.com/) container, spawn from a provided image. To create a local instance of the image, named `tasmania`, from the repository root directory issue:

	docker build --build-arg uid=$(id -u) -t tasmania .

Later, launch

	docker run --rm -dit --privileged -e DISPLAY -e XAUTHORITY=$XAUTORITY -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/tasmania tasmania

to run the container in detached mode. The flag `-v $PWD:/tasmania` attaches the current directory to the container *volume*, so that the folder :option:`/tasmania` within the container will be in sync with the repository root directory. Instead, the flags `--privileged -e DISPLAY -e XAUTHORITY:$XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix` are required to execute any graphical application within the container, and should then be used to, e.g., generate a plot via [Matplotlib](https://matplotlib.org/) or run a [Jupyter](http://jupyter.org/) notebook. **Note**: we are conscious of the fact that granting a container privileged rights is consider bad practice. Yet, it is the easiest way (to our knowledge) to allow containerized applications access the host desktop environment.

When executed successfully, the previous command returns a long sequence of letters and digits, representing the container identifier. This is required to gain shell access to the running container:

	docker exec -it CONTAINER_ID bash

Here, `CONTAINER_ID` is the container identifier.
