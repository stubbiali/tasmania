API Documentation
=================

Axis
----

.. autoclass:: grids.axis.Axis
   :members:  


Dynamics
--------

Diagnostics
^^^^^^^^^^^

.. autoclass:: dycore.diagnostic_isentropic.DiagnosticIsentropic
   :members:

Dynamical cores
^^^^^^^^^^^^^^^

.. autoclass:: dycore.dycore.DynamicalCore
   :members:

.. autoclass:: dycore.dycore_isentropic.DynamicalCoreIsentropic
   :members:

Lateral boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: dycore.horizontal_boundary.HorizontalBoundary
   :members:

.. autoclass:: dycore.horizontal_boundary.Periodic
   :members:

.. autoclass:: dycore.horizontal_boundary.PeriodicXZ
   :members:

.. autoclass:: dycore.horizontal_boundary.PeriodicYZ
   :members:

.. autoclass:: dycore.horizontal_boundary.Relaxed
   :members:

.. autoclass:: dycore.horizontal_boundary.RelaxedSymmetricXZ
   :members:

.. autoclass:: dycore.horizontal_boundary.RelaxedSymmetricYZ
   :members:

Horizontal smoothing
^^^^^^^^^^^^^^^^^^^

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothing
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderXYZ
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderXZ
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderYZ
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderXYZ
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderXZ
   :members:

.. autoclass:: dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderYZ
   :members:

Numerical fluxes
^^^^^^^^^^^^^^^^

.. autoclass:: dycore.flux_isentropic.FluxIsentropic
   :members:

.. autoclass:: dycore.flux_isentropic_upwind.FluxIsentropicUpwind
   :members:

.. autoclass:: dycore.flux_isentropic_centered.FluxIsentropicCentered
   :members:

.. autoclass:: dycore.flux_isentropic_maccormack.FluxIsentropicMacCormack
   :members:

Prognostics
^^^^^^^^^^^

.. autoclass:: dycore.prognostic_isentropic.PrognosticIsentropic
   :members:

.. autoclass:: dycore.prognostic_isentropic_forward_euler.PrognosticIsentropicForwardEuler
   :members:

.. autoclass:: dycore.prognostic_isentropic_centered.PrognosticIsentropicCentered
   :members:

Wave absorber
^^^^^^^^^^^^^

.. autoclass:: dycore.vertical_damping.VerticalDamping
   :members:

.. autoclass:: dycore.vertical_damping.VerticalDampingRayleigh
   :members:


Grids
-----

Two-dimensional grids
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: grids.grid_xy.GridXY
   :members:

.. autoclass:: grids.grid_xz.GridXZ
   :members:

.. autoclass:: grids.sigma.Sigma2d
   :members:

.. autoclass:: grids.gal_chen.GalChen2d
   :members:

.. autoclass:: grids.sleve.SLEVE2d
   :members:

Three-dimensional grids
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: grids.grid_xyz.GridXYZ
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


Parameterizations
-----------------

.. autoclass:: parameterizations.adjustment.Adjustment
   :members:

.. autoclass:: parameterizations.tendency.Tendency
   :members:

Microphysics
^^^^^^^^^^^^

.. autoclass:: parameterizations.adjustment_microphysics.AdjustmentMicrophysics
   :members:

.. autoclass:: parameterizations.adjustment_microphysics.AdjustmentMicrophysicsKessler
   :members:


Storages
--------

.. autoclass:: storages.grid_data.GridData
   :members:

.. autoclass:: storages.state_isentropic.StateIsentropic
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

.. automodule:: utils.utils
   :members:

Meteo utilities
^^^^^^^^^^^^^^^

.. automodule:: utils.utils_meteo
   :members:

Plotting utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: utils.utils_plot
   :members:
