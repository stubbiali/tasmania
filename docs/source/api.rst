API Documentation
=================

Axis
----

.. autoclass:: grids.axis.Axis
   :members:  


Dynamics
--------------

Diagnostics
^^^^^^^^^^^

.. autoclass:: dycore.isentropic_diagnostic.IsentropicDiagnostic
   :members:

Dynamical cores
^^^^^^^^^^^^^^^

.. autoclass:: dycore.dycore.DynamicalCore
   :members:

.. autoclass:: dycore.dycore.IsentropicDynamicalCore
   :members:

Lateral boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dycore.horizontal_boundary.HorizontalBoundary
   :members:

.. autoclass:: dycore.horizontal_boundary.Periodic
   :members:

.. autoclass:: dycore.horizontal_boundary.Relaxed
   :members:

.. autoclass:: dycore.horizontal_boundary.RelaxedSymmetricXZ
   :members:

.. autoclass:: dycore.horizontal_boundary.RelaxedSymmetricYZ
   :members:

Numerical diffusion
^^^^^^^^^^^^^^^^^^^

.. autoclass:: dycore.diffusion.Diffusion
   :members:

Numerical fluxes
^^^^^^^^^^^^^^^^

.. autoclass:: dycore.isentropic_flux.IsentropicFlux
   :members:

.. autoclass:: dycore.isentropic_flux.UpwindIsentropicFlux
   :members:

.. autoclass:: dycore.isentropic_flux.LeapfrogIsentropicFlux
   :members:

.. autoclass:: dycore.isentropic_flux.MacCormackIsentropicFlux
   :members:

Prognostics
^^^^^^^^^^^

.. autoclass:: dycore.isentropic_prognostic.IsentropicPrognostic
   :members:

.. autoclass:: dycore.isentropic_prognostic.OneTimeLevelIsentropicPrognostic
   :members:

.. autoclass:: dycore.isentropic_prognostic.TwoTimeLevelsIsentropicPrognostic
   :members:

Wave absorber
^^^^^^^^^^^^^

.. autoclass:: dycore.vertical_damping.VerticalDamping
   :members:

.. autoclass:: dycore.vertical_damping.Rayleigh
   :members:


Grids
-----

Two-dimensional grids
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: grids.xy_grid.XYGrid
   :members:

.. autoclass:: grids.xz_grid.XZGrid
   :members:

.. autoclass:: grids.sigma.Sigma2d
   :members:

.. autoclass:: grids.gal_chen.GalChen2d
   :members:

.. autoclass:: grids.sleve.SLEVE2d
   :members:

Three-dimensional grids
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: grids.xyz_grid.XYZGrid
   :members:

.. autoclass:: grids.sigma.Sigma3d
   :members:

.. autoclass:: grids.gal_chen.GalChen3d
   :members:

.. autoclass:: grids.sleve.SLEVE3d
   :members:


Model
-----

.. autoclass:: model.Model
   :members:


Namelist
--------

.. automodule:: namelist
   :members:


Storages
--------

.. autoclass:: storages.grid_data.GridData
   :members:

.. autoclass:: storages.isentropic_state.IsentropicState
   :members:


Topography
----------

.. automodule:: grids.topography
   :members:

Parsers
^^^^^^^

.. autoclass:: grids.parser.parser_1d.Parser1d
   :members: __cinit__, evaluate

.. autoclass:: grids.parser.parser_2d.Parser2d
   :members: __cinit__, evaluate


Utilities
---------

.. automodule:: utils
   :members:

