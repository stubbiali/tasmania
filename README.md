A Python library for Earth system science
===================================================

`Tasmania` aims to provide domain scientists with a high-level, highly-modular and flexible framework to define, simulate and assess finite difference Earth system science models, e.g., models for climate and weather forecasting. The library leverages on `GridTools4Py`, a Python interface to the C++ template library `GridTools`, developed at ETH/CSCS. `GridTools` furnishes a wide gamma of tools to implement stencil-based operations, hence it finds a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a particular architecture, e.g., Xeon Phi or GPU. In addition to these, `GridTools4Py` supplies some Pythonic backends suitable for debugging and research purposes. Nevertheless, `GridTools`'s frontend, hence `GridTools4Py`'s API, is hardware-agnostic, implying that user's code does not need to be changed when porting it to a new architecture. This enables a complete separation of concerns between domain scientists - who can work in a familiar development environment like Python - and computer scientists - who oversee the translation and compilation stage. 


Tasmania's virtual environment
------------------------------

In order to setup a suitable virtual environment where to run any `Tasmania`-related code, please pursue the following steps:

1. Install `VirtualBox`, available at https://www.virtualbox.org/.
2. Install `vagrant` from terminal by

	apt-get install vagrant

   Admin privileges may be required.
3. From the repo root directory, type

	vagrant up --provision
		 
   This command will create a virtual machine (VM) in your system, so it may take some time (even in the order of hours).
   
Thereafter:

1. To boot and login to the VM, from the repo root directory type:

	vagrant up
	vagrant ssh

2. Within the VM, you can switch the virtual environment on by

	source venv/bin/activate

   This command should be invoked at the beginning of each session, so you may want to add it to `~/.bashrc`. 
3. The first time you login to the VM, move to the folder `tasmania`, then write

	make

   This will compile some auxiliary C++ code and create the documentation (later available at `docs/build/`).
4. Finally, `CTRL + D` to logout, then

	vagrant halt

   to shut down the VM.
