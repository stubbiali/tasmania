<img align="right" src="taz.jpeg">

Tasmania
==

This is the repository for Tasmania, a Python framework/toolkit to ease the composition, configuration, and execution of Earth system models.

Background and motivation
--

Weather and climate models are complex systems comprising several subsystems (atmosphere, ocean, land, glacier, sea ice, and marine biogeochemistry) which interact at their interfaces through the exchange of mass, momentum and energy. Each domain hosts a plethora of interlinked physical and chemical processes which can be characterized on a wide spectrum of spatio-temporal scales. Due to limited available computer resources, creating a discrete model covering the entire range of scales, even for a single subsystem, is challenging, if not an impossible task. Rather, the time step and the grid size discriminate between fully resolved fluid-dynamics features (e.g., horizontal and vertical advection, pressure gradient, and Coriolis effects), and subgrid-scale aspects (e.g., radiative transfer, macro- and micro-physics, shallow and deep convection, turbulent mixing in the planetary boundary layer, and orographic drag) which do not emerge naturally on the grid. The former are traditionally referred to as the *dynamics*, while the latter form the so-called *physics* of the model.

The focus of this thesis is on atmospheric models, where a *dynamical core* solves for the fluid-dynamics equations and *physical parameterizations* express the bulk effect of subgrid-scale phenomena upon the large-scale dynamics in terms of the resolved fields. The procedure which molds all the dynamical and physical components to yield a coherent and comprehensive model is referred to as the *physics-dynamics coupling*. 

For the sake of tractability, individual groups and institutions develop dynamical kernels and parameterizations in isolation, working independently one from the other. This approach helps individuals acquire a deep understanding of single processes and develop sound numerics for individual components. In contrast, the physics-dynamics coupling has historically received less attention. As a result, it is poorly understood and components are commonly tied in a crude and low-accurate fashion. As the error associated with the discretization of individual processes decreases, the error injected by the coupling will eventually dominate. To a certain extent, this deficiency may be ascribed to the lack of flexibility, interoperability, and usability of traditional frameworks.

Goal
--

Tasmania aims to be a high-level, modular and flexible framework to ease the composition, configuration, simulation and evaluation of finite difference numerical schemes for Earth system science. The framework features a component-based architecture, with each component being a Python class representing a dynamical or physical process. As a result, the user is given fine-grained control on the execution flow.
	
Fact sheet
--

 - Physical components must conform to [sympl](https://github.com/mcgibbon/sympl)'s (System for Modelling Planets) primitives application programming interface (API). 
 
 - To facilitate the development of dynamical kernels, \texttt{tasmania} provides an abstract base class (ABC) with intended support for multi-stage time-integrators (e.g., Runge-Kutta schemes) and partial operator splitting techniques, which integrate slow and fast processes with large and multiple small time steps, respectively. To this end, a distinction between \emph{slow} physics (calculated over the large time step, outside of the dynamical core), \emph{intermediate} physics (evaluated over the large time step at every stage) and \emph{fast} physics (computed over the shorter time step at each sub-step) is made. 
 
 - The following coupling mechanisms are currently implemented:
 
	 - concurrent coupling;
	 - parallel splitting (*not working, sorry...*);
	 - sequential-splitting; 
	 - symmetrized sequential-splitting.
	 
	 Hybrid approaches are possible. 
 - A simplified hydrostatic model in isentropic coordinates is available as proof-of-concept. Finite difference operators arising from the numerical discretization of the model are implemented via GridTools4Py.

GridTools and GridTools4Py
--

[GridTools4Py](https://github.com/eth-cscs/gridtools4py) is a domain specific language (DSL) for stencil-based codes. It offers a high-level entry point to the C++ template library [GridTools](https://github.com/eth-cscs/gridtools). Both tools have been developed at ETH/CSCS. 

GridTools furnishes a wide gamma of tools to implement stencil-based operations, thus finding a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., x86, MIC, and GPU. In addition to these, GridTools4Py supplies some Pythonic back-ends suitable for debugging and research purposes. 

Conversely, GridTools's front-end, then GridTools4Py's interface, are hardware-agnostic, so that the user's code can be left unchanged when porting it to different architectures. This enables a complete separation of concerns between domain scientists - who can work in a familiar and powerful development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

Installation and usage
--

To clone this repository on your machine and place yourself on the current branch, from within a terminal run

	git clone https://github.com/eth-cscs/tasmania.git

We suggest to run any script or application leveraging Tasmania inside a [Docker](https://www.docker.com/) container, spawn from a provided image. To clone the GridTools4Py repository and create a local instance of the image (named `tasmania:master`), from the repository root directory issue

	make docker-build

and follow the on-screen instructions.

**Note**: both Tasmania and GridTools4Py repositories are *private*, so you should be given access to them in order to accomplish the steps above.

Later, launch

	make docker-run

to run a container in the background and get shell access to it. This command only works on Linux systems; on a Mac OS X, the equivalent command is

	make docker-run-mac

You login the container's shell as `tasmania-user` - a non-root user having the same `UID` of the host user who spun up the container. The container's working directory is `/home/tasmania-user`.

**Disclaimer**: the container is given extended privileges to be able to run graphical applications, like generating a plot via [Matplotlib](https://matplotlib.org/) or running a [Jupyter](http://jupyter.org/) notebook. We are conscious of the fact that this is consider bad practice. Yet, it is the easiest way (to our knowledge) to allow containerized applications access the host desktop environment.

**Remark**: ax explained [here](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc), running a graphical application inside a Docker container on a Mac OS X requires *socat* and *Xquartz* to let the container harness the X window system on the host operating system. Both socat and Xquartz can be installed from the command line through *homebrew*:

	brew install socat
	brew install xquartz

Except for `tasmania/cpp`, any other file and folder present in the local repository is bind mounted into the container under `/home/tasmania-user/tasmania`. This ensures that the latest version of Tasmania is always available inside the container. More generally, data in the repository can be seamlessly shared between the container and the host.

Inside the container, the following isolated Python environments, created via [virtualenv](https://virtualenv.pypa.io/en/latest/), are available:

 - `/home/tasmania-user/py35`, built around the Python3.5 interpreter;
 - `/home/tasmania-user/py36`, built around the Python3.6 interpreter;
 - `/home/tasmania-user/py37`, built around the Python3.7 interpreter.
 
 To activate the virtual environment `py3x`, execute the binary `py3x/bin/activate`. To exit the virtual environment, from any location issue `deactivate`.

`Ctrl+D` to exit the container's shell. To gain back shell access to the running container:

	docker exec -it CONTAINER_ID bash

`CONTAINER_ID` is the container identifier; it can be retrieved by inspecting the output of

	docker ps -a

Eventually, you can remove the container via

	docker stop CONTAINER_ID

Running the tests
--

To run the whole test suite against the Python3.x interpreter using [pytest](https://docs.pytest.org/en/latest/), from within the container working directory run:
```bash
tasmania-user@CONTAINER_ID:~$ . py3x/bin/activate
(py3x) tasmania-user@CONTAINER_ID:~$ cd tasmania
(py3x) tasmania-user@CONTAINER_ID:~/tasmania$ make prepare-tests-py3x && make tests
```

Repository directory structure
--

- `buffer/`: convenient location for files (e.g. Matplotlib figures) generated inside the container and to be moved to other host's directory.
- `docker/`: configuration files and scripts to create a Docker image and run a Docker container.
- `docs/`: [Sphinx](http://www.sphinx-doc.org/en/master/) documentation.
- `drivers/`: namelists and drivers.
- `notebooks/`: Jupyter notebooks.
- `results/`: figures (`figures/`) and animations (`movies/`) generated via Matplotlib.
- `scripts/`: bash (`bash/`), Python (`python/`), and Slurm (`slurm/`) scripts for, e.g., post-processing, plotting, sharing data with a remote machine.
- `spike/`: miscellaneous of old (i.e. deprecated), experimental, and potentially useful stuff.
- `tasmania/`: codebase, consiting of Python (`python/`) and C++ (`cpp`) source files.
- `tests/`: test suite. 

Makefile targets
--

- `docker-build`: builds the image `tasmania:base` against the dockerfile `docker/dockerfiles/dockerfile.base` and dumps it in a tar archive, clones the GridTools4Py repository, creates a minimal version of this repository under `docker/tasmania`, and builds the image `tasmania:master` against the dockerfile `docker/dockerfiles/dockerfile.tasmania` and dumps it in a tar archive. Any of these steps is optional and can be skipped.
- `docker-run`, `docker-run-mac`: runs and connects to a container spawn from the image `tasmania:master`.
- `docs`: builds the documentation for Tasmania via Sphinx in HTML, LaTeX and LaTeX-pdf format.  
- `prepare-tests-py35`,`prepare-tests-py36`,`prepare-tests-py37`: generate the baseline images needed by the tests. 
- `tests`: runs the tests.
- `clean`: deletes temporary, unnecessary files.
- `distclean`: as `clean`, but deletes *necessary* binary files and the built documentation as well. Use it carefully!
- `gitignore`: adds too large files to `.gitignore`.
