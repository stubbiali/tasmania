[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<img align="right" src="taz.jpeg">

# Tasmania

This is the repository for Tasmania, a Python framework/toolkit to ease the composition, configuration, simulation and monitoring of Earth system models.

## Background and motivation

Weather and climate models are complex systems comprising several subsystems (atmosphere, ocean, land, glacier, sea ice, and marine biogeochemistry) which interact at their interfaces through the exchange of mass, momentum and energy. Each domain hosts a plethora of interlinked physical and chemical processes which can be characterized on a wide spectrum of spatio-temporal scales. Due to the limited available computer resources, creating a discrete model covering the entire range of scales, even for a single subsystem, is challenging, if not an impossible task. Rather, the grid resolution discriminates between fully resolved fluid-dynamics features (e.g., horizontal and vertical advection, pressure gradient, and Coriolis acceleration), and subgrid-scale aspects (e.g., radiative transfer, macro- and micro-physics, shallow and deep convection, turbulent mixing in the planetary boundary layer, and orographic drag) which do not emerge naturally on the mesh. The former are traditionally referred to as the *dynamics*, while the latter form the so-called *physics* of the model.

In all models, the *dynamical core* solves for the fluid-dynamics equations while *physical parameterizations* express the bulk effect of the subgrid-scale phenomena upon the large-scale flow. The procedure which molds all the dynamics and physics components to yield a coherent and comprehensive model is referred to as the *physics-dynamics coupling*. 

The continual growth in model resolution demands for increasing specialization to address the physical processes which emerge on smaller and smaller scales. This has resulted in a high compartmentalization of the model development, with dynamical cores and physics packages mostly developed in isolation. Besides easing the proliferation of software components with incompatible structure, such approach is in direct contrast with the need of improving the time stepping in the current apparatus of atmospheric models. Indeed, the time stepping is often merely accurate to the first order. However, as the error associated with the discretization of individual processes decreases, the error injected by the coupling will eventually dominate. 

## Goal

Tasmania aims to provide a high-level platform to aid the investigation of the physics-dynamics coupling in atmospheric models. The framework features a component-based architecture, in which each component represents a dynamical or physical process. Couplers are offered which chain individual components pursuing a well-defined coupling algorithm.
	
## Fact sheet

 - Physical components must conform to [sympl](https://github.com/stubbiali/sympl)'s (System for Modelling Planets) primitives API. 
 
 - To facilitate the development of dynamical kernels, Tasmania provides an abstract base class (ABC) with intended support for multi-stage time-integrators (e.g., Runge-Kutta schemes) and partial operator splitting techniques, which integrate slow and fast processes with large and multiple small time steps, respectively. To this end, a distinction between *slow* physics (calculated over the large time step, outside of the dynamical core), *intermediate* physics (evaluated over the large time step at every stage) and *fast* physics (computed over the shorter time step at each sub-step) is made. 
 
 - The following coupling mechanisms are currently implemented:
 
	 - full coupling;
	 - *lazy* full coupling;
	 - parallel splitting;
	 - sequential-tendency splitting;
	 - sequential-update splitting; 
	 - symmetrized sequential-update splitting, or Strang splitting.
	 
	 Hybrid approaches are possible. 
 - The two-dimensional viscous Burgers' equations and a simplified hydrostatic model in isentropic coordinates have been coded as proof-of-concepts. Finite difference operators arising from the numerical discretization of the model are implemented via GT4Py.

## GridTools and GT4Py

[GT4Py](https://github.com/GridTools/gt4py) is a domain specific language (DSL) for stencil-based codes. It offers a high-level entry point to the C++ template library [GridTools](https://github.com/eth-cscs/gridtools). Both tools have been developed at ETH/CSCS in collaboration with MeteoSwiss. 

GridTools furnishes a wide gamma of tools to implement stencil-based operations, thus finding a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., x86, MIC, and GPU. In addition to these, GT4Py supplies some Pythonic back-ends suitable for debugging and research purposes. 

Conversely, GridTools's front-end, then GT4Py's interface, are hardware-agnostic, so that the user's code can be left unchanged when porting it to different architectures. This enables a complete separation of concerns between domain scientists - who can work in a familiar and powerful development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

## Installation

To clone this repository (with submodules) on your machine and place yourself on the current branch, run

	git clone -b master --recurse-submodules https://github.com/eth-cscs/tasmania.git

**Note:** both Tasmania and GT4Py repositories are *private*, so you should be granted access to clone them and accomplish all the actions listed below.
	
## Running Tasmania

### Using a virtual environment

The `bootstrap_venv.sh` bash script automates the creation of an isolated Python environment for Tasmania. The [virtualenv](https://virtualenv.pypa.io/en/latest/) package is used. The procedure can be customized by means of the bash variables defined in the preamble of the script. 

### Using a Docker container on a local machine

We provide two [Docker](https://www.docker.com/) images to run a containerized version of Tasmania. The two images are based on `ubuntu:18.04` and `nvidia/cuda:latest`. Please note that the former image does not contain the CUDA toolkit, so it only supports the CPU-based backends of GT4Py. To create a local instance of the images (named `tasmania:cpu` and `tasmania:gpu`, respectively), from the root directory of the repository issue

	make docker-build-cpu
	make docker-build-gpu

and follow the on-screen instructions. Later, launch

	make docker-run-cpu
	
or

    make docker-run-gpu

to run a container in the background and get shell access to it. You login the container's shell as `tasmania-user` - a non-root user having the same `UID` of the host user who spun up the container. The container's working directory is `/home/tasmania-user`.

**Disclaimer:** the container is given extended privileges to be able to run graphical applications, like generating a plot via [Matplotlib](https://matplotlib.org/) or running a [Jupyter](http://jupyter.org/) notebook. We are conscious of the fact that this is consider bad practice. Yet, it is the easiest way (to our knowledge) to allow containerized applications access the host desktop environment.

**Remark:** ax explained [here](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc), running a graphical application inside a Docker container on a Mac OS X requires *socat* and *Xquartz* to let the container harness the X window system on the host operating system. Both socat and Xquartz can be installed from the command line through, e.g., *homebrew*:

	brew install socat
	brew install xquartz

Except for `tasmania/cpp`, any other file and folder present in the local repository is bind mounted into the container under `/home/tasmania-user/tasmania`. This ensures that the latest version of Tasmania is always available inside the container. More generally, data in the repository are thus seamlessly shared between the container and the host.

Inside the container, the following isolated Python environments, created via virtualenv, are available:

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
	
### Using a Docker container on Piz Daint

Several aspects make Docker unsuitable to fit the needs of high-performance computing (HPC). Different ongoing initiatives are attempting to put the recognized power of containers at the service of HPC users; notable examples include Shifter, Singularity and [Sarus](http://hpcadvisorycouncil.com/events/2019/swiss-workshop/pdf/030419/K_Mariotti_CSCS_SARUS_OCI_ContainerRuntime_04032019.pdf). In the following, we detail a simple workflow harnessing Sarus to run Tasmania-based applications on the [Piz Daint](https://www.cscs.ch/computers/piz-daint/) supercomputer housed at CSCS. The workflow is tailored on the filesystem and programming ecosystem of Piz Daint. However, the logical steps should be easily reproducible on other HPC platforms upon small modifications to the scripts. 

To set up a favourable environment for Tasmania:
1. On your local machine, save the Docker image to a tar archive by running `make docker-build-cpu` or `make docker-build-gpu` and following the on-screen instructions;
2. Copy your local instance of the repository on Piz Daint under, e.g., `$PROJECT`. You may want to use the customizable bash script `scripts/bash/transfer_to_remote.sh`;
3. On Piz Daint, copy the content of `scripts/scratch_daint/` in `$SCRATCH`. From now on, all commands are supposed to be issued from `$SCRATCH`;
4. Load the Docker image by submitting the Batch script `sarus_load.run` via Slurm.

Once done, you can:
1. Obtain an interactive job allocation via `salloc.sh` and then spin up a container on the compute node via `sarus_run.sh`. 
2. Submit a job running a simulation by using the Batch script `sarus_python_driver.run`.
3. Submit a job executing a Python script by using the Batch script `sarus_python_script.run`.
4. Submit a job executing a bash script by using the Batch script `sarus_bash_script.run`.

**Remark:** All the aforementioned scripts are customizable to meet user-specific needs. 

**Another remark:** Inside a container spawn on Piz Daint, the user is exposed to a folder organization identical to that described in the previous section. This should enable a smooth user experience when migrating from your own local machine to the remote server. The containerized directory `/home/tasmania-user/tasmania` is mapped to an instance of the repository present on Piz Daint, e.g., under `$PROJECT`. This is not valid for `/home/tasmania-user/tasmania/data` which is rather mapped to the `$SCRATCH/buffer` directory on the host to ensure it is write-accessible. Indeed, any output produced inside the container should be stored under `/home/tasmania-user/tasmania/data`. If it does not exist, the folder `$SCRATCH/buffer` gets created by either `sarus_run.sh`, `sarus_python_driver.run` or `sarus_python_script.run`.

## Running the benchmarks

The benchmarks for the Burgers' equations and the isentropic model are stored in the `drivers/` folder. Please see the README therein for further details.

## Repository directory structure

- `buffer/`: convenient location for files (e.g. Matplotlib figures) generated inside the container and to be moved to other host's directory.
- `data/`: data sets;
- `docker/`: configuration files and scripts to create the Docker images and run Docker containers.
- `docker/external`: the GT4Py (`gt4py/`) and sympl (`sympl/`) submodules.
- `docs/`: [Sphinx](http://www.sphinx-doc.org/en/master/) documentation (out-of-date).
- `drivers/`: drivers and associated namelist files to run the Burgers' model (`burgers/`) or the isentropic model (`isentropic/`).
- `notebooks/`: Jupyter notebooks.
- `results/`: figures (`figures/`) and animations (`movies/`) generated via Matplotlib.
- `scripts/`: bash (`bash/`), Python (`python/`), Slurm (`slurm/`) scripts for, e.g., post-processing, plotting, sharing data with a remote machine. `scratch_daint/` and `scratch_dom/` contain files which should be copy to the `$SCRATCH` folder on Piz Daint and Dom, respectively.
- `tasmania/`: codebase, consisting of Python (`python/`) and C++ (`cpp/`) source files.
- `tests/`: test suite. 

## Makefile targets

- `docker-build-cpu`: builds the image `tasmania:base-cpu` against the dockerfile `docker/cpu/dockerfiles/dockerfile.base` and the image `tasmania:cpu` against the dockerfile `docker/cpu/dockerfiles/dockerfile.tasmania`.
- `docker-build-gpu`: builds the image `tasmania:base-gpu` against the dockerfile `docker/gpu/dockerfiles/dockerfile.base` and the image `tasmania:gpu` against the dockerfile `docker/gpu/dockerfiles/dockerfile.tasmania`.
- `docker-run-cpu`: runs and connects to a container spawn from the image `tasmania:cpu`.
- `docker-run-gpu`: runs and connects to a container spawn from the image `tasmania:gpu`.
- `docs`: builds the documentation for Tasmania via Sphinx in HTML, LaTeX and LaTeX-pdf format.  
- `tests`: prepares and runs the tests.
- `clean`: deletes temporary, unnecessary files.
- `distclean`: as `clean`, but deletes *necessary* binary files and the built documentation as well. Use it carefully!
- `gitignore`: adds too large files to `.gitignore`.
