<img align="right" src="taz.jpeg">

A Python library for Earth science
==================================

`Tasmania` aims to be a high-level, highly-modular and flexible framework to compose, simulate and evaluate finite difference numerical schemes for Earth system science. Relevant examples are climate and weather forecasting models. The library leverages `GridTools4Py`, a complete set of Python bindings for the C++ template library `GridTools`, developed at ETH/CSCS. `GridTools` furnishes a wide gamma of tools to implement stencil-based operations. Thus, it finds a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., Xeon Phi or GPU. In addition to these, `GridTools4Py` supplies some Pythonic backends suitable for debugging and research purposes. 

On the other hand, `GridTools`'s frontend, hence `GridTools4Py`'s interface, is hardware-agnostic, implying that user's codebase does not need to be changed when ported to a different architecture. This enables a complete separation of concerns between domain scientists - who can work in a familiar development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

Set up a Docker container for `Tasmania`
----------------------------------------

A *container* is a runtime instance of an *image*, which in turn is an executable package that includes everything needed to run an application. As opposed to a *virtual machine* (VM), a container runs natively on Linux and shares the kernel of the host machine with other containers, thus resulting in a lightweight process. This motivates the large interest containers have gained in the last years.  

[Docker](https://www.docker.com/) is a platform for developers and sysadmins to develop, deploy, and run applications with containers. To create a local instance of the Docker image from which the container for `Tasmania` will be spawn, from the repo root directory issue:

	docker build -t tasmania docker

Later, type

	docker run -dit -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $PWD:/tasmania tasmania

to run the container in detached mode. The flags `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` are required only if ones want to generate plots via [Matplotlib](https://matplotlib.org/). If this is not the case, they can be omitted. The flag `-v $PWD:/tasmania` attaches the current directory to the container *volume*. In this way, the folder `/tasmania` within the container will be in sync with the repo root directory.

When successful, the previous command returns a long sequence of letters and digits, representing the container identifier. This is required to gain shell access to the running container:

	docker exec -it CONTAINER_ID bash

Here, `CONTAINER_ID` is the container identifier.
