============
Introduction
============

Motivation
==========

Available computer resources limit the resolution of climate projection and weather forecasting models, and thus pose a strict lower bound on the smallest scale of motion which can be directly resolved. Processes taking place at spatial scales smaller than the grid size, or occurring on timescales shorter than the model timestep, can not be explicitly represented. Notable examples include turbulent fluxes in the planetary boundary layer, radiative transfer, (shallow) convection, and cloud microphysics. As in many situations these processes exert a strong influence on the large-scale fields, they cannot simply be neglected. Rather, subgrid-scale phenomena are *parameterized*, i.e., their effect is expressed in terms of the resolved variables.

Although physical parameterizations have been largely studied in isolation, their combined effect, and their interaction with the underlying dynamical core (traditionally referred to as *physics-dynamics coupling*), has received significantly less attention. Among the pletora of motivations behind this deficiency, the lack of flexibility in most legacy codes, making them unsuitable to address this kind of questions, has been playing a prominent role. 

Objective
=========

**Tasmania** aims to be a high-level, modular and flexible framework to ease the composition, configuration, simulation and evaluation of finite difference numerical schemes for Earth system science. The library leverages Sympl_ and GridTools4Py_. The former provides the base classes and array and constants handling functionality. The latter is a complete set of Python bindings for the C++ template library GridTools_, developed at ETH/CSCS. GridTools furnishes a wide gamma of tools to implement stencil-based operators, thus finding a natural application in finite difference codes. It ships with different lower-level and high-performance backends, each one designed for a specific architecture, e.g., x86, MIC, and GPU. In addition to these, GridTools4Py supplies some Pythonic backends suitable for debugging and research purposes. On the other hand, GridTools's frontend, then GridTools4Py's interface, are hardware-agnostic, so that the user's code can be left unchanged when porting it to different architectures. This enables a complete separation of concerns between domain scientists - who can work in a familiar and powerful development environment like Python - and computer scientists - who oversee the translation, compilation and execution stage. 

.. _Sympl: https://github.com/mcgibbon/sympl
.. _GridTools4Py: https://github.com/eth-cscs/gridtools4py
.. _GridTools: https://github.com/eth-cscs/gridtools
